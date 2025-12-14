from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, date, timedelta
import json
import logging
import os
from pathlib import Path
from collections import defaultdict, Counter
import re
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
from groq import Groq
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Reddit Insights API",
    description="Optimized API for Reddit analytics with comprehensive dashboard endpoints",
    version="5.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# Read credentials
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_FILE_PATH = Path(__file__).parent.parent / "data" / "data.jsonl"

# Initialize Groq client
try:
    client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    client = None

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)


# ------------------ MODELS ------------------

class DateRange(BaseModel):
    start_date: date
    end_date: date

    @validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

class PlatformRequest(BaseModel):
    platform: str = Field(..., pattern="^(youtube|reddit|both)$")
    date_range: DateRange

# Dashboard Response Models
class PlatformDistribution(BaseModel):
    platform: str
    posts_count: int
    percentage: float

class EngagementTrend(BaseModel):
    month: str
    reddit_engagement: int
    total_engagement: int

class TopPost(BaseModel):
    rank: int
    title: str
    content: str
    engagement_score: int
    platform: str
    url: str
    author: str
    created_date: str

class InfluentialUser(BaseModel):
    rank: int
    username: str
    total_engagement: int
    posts_count: int
    avg_engagement: float
    platform: str

class TrendingKeyword(BaseModel):
    keyword: str
    frequency: int
    posts_count: int

class PostsHistogram(BaseModel):
    time_period: str
    reddit_posts: int
    total_posts: int

class TrendingComment(BaseModel):
    rank: int
    author: str
    content: str
    engagement_score: int
    platform: str
    url: str
    post_title: str

class SentimentPoint(BaseModel):
    month: str
    positive_score: float
    negative_score: float
    neutral_score: float

class SentimentTrend(BaseModel):
    reddit: List[SentimentPoint]

# Main Dashboard Response
class DashboardResponse(BaseModel):
    total_posts: int
    total_engagement: int
    influential_users_count: int
    trending_keywords_count: int
    platform_distribution: List[PlatformDistribution]
    engagement_trends: List[EngagementTrend]
    top_posts: List[TopPost]
    influential_users: List[InfluentialUser]
    trending_keywords: List[TrendingKeyword]
    posts_histogram: List[PostsHistogram]
    trending_comments: List[TrendingComment]
    sentiment_trends: SentimentTrend


# ------------------ DATA LOADING AND CACHING ------------------

@lru_cache(maxsize=1)
def load_reddit_data() -> List[Dict]:
    """Load Reddit data from JSONL file with caching."""
    data = []
    
    if not DATA_FILE_PATH.exists():
        logger.error(f"Data file not found at {DATA_FILE_PATH}")
        return []
    
    try:
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        record = json.loads(line)
                        
                        # Extract nested data structure
                        if 'kind' in record and 'data' in record:
                            post_data = record['data']
                            post_data['_kind'] = record['kind']
                            post_data['_id'] = post_data.get('id')
                            
                            # Handle both top-level and nested data structures
                            if 'data' in post_data:
                                nested_data = post_data['data']
                                nested_data.update({k: v for k, v in post_data.items() if k != 'data'})
                                data.append(nested_data)
                            else:
                                data.append(post_data)
                        
                        if line_num % 100 == 0:
                            logger.debug(f"Loaded {line_num} lines...")
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error at line {line_num}: {e}")
                        continue
        
        logger.info(f"Successfully loaded {len(data)} Reddit records")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return []

def is_post(item: Dict) -> bool:
    """Check if item is a post (not a comment)."""
    kind = item.get('_kind', '')
    return kind == 't3' or 'title' in item

def is_comment(item: Dict) -> bool:
    """Check if item is a comment."""
    kind = item.get('_kind', '')
    return kind == 't1' or item.get('link_id') is not None

def parse_reddit_timestamp(timestamp: Any) -> Optional[date]:
    """Parse Reddit timestamp to date."""
    if timestamp is None:
        return None
    
    try:
        if isinstance(timestamp, (int, float)):
            # Convert Unix timestamp to date
            return datetime.fromtimestamp(timestamp).date()
        elif isinstance(timestamp, str):
            if timestamp.replace('.', '', 1).isdigit():
                return datetime.fromtimestamp(float(timestamp)).date()
        return None
    except Exception:
        return None

def get_post_engagement(post: Dict) -> int:
    """Calculate engagement score for a Reddit post."""
    score = post.get('score', 0)
    if not isinstance(score, (int, float)):
        try:
            score = int(score)
        except:
            score = 0
    
    # Add comments as engagement
    num_comments = post.get('num_comments', 0)
    if not isinstance(num_comments, (int, float)):
        try:
            num_comments = int(num_comments)
        except:
            num_comments = 0
    
    # Engagement formula: score + (comments * 2)
    engagement = max(1, score + (num_comments * 2))
    return engagement

def get_comment_engagement(comment: Dict) -> int:
    """Calculate engagement score for a Reddit comment."""
    score = comment.get('score', 0)
    if not isinstance(score, (int, float)):
        try:
            score = int(score)
        except:
            score = 0
    
    return max(1, score)

def get_post_url(post: Dict) -> str:
    """Get Reddit post URL."""
    permalink = post.get('permalink')
    if permalink:
        return f"https://www.reddit.com{permalink}"
    elif post.get('url'):
        return post.get('url')
    elif post.get('_id'):
        return f"https://www.reddit.com/comments/{post.get('_id')}"
    return "#"

def get_comment_url(comment: Dict) -> str:
    """Get Reddit comment URL."""
    link_id = comment.get('link_id', '')
    comment_id = comment.get('_id', '')
    if link_id and comment_id:
        # Remove 't3_' prefix from link_id if present
        if link_id.startswith('t3_'):
            post_id = link_id[3:]
        else:
            post_id = link_id
        return f"https://www.reddit.com/comments/{post_id}/_/{comment_id}"
    return "#"

def get_post_content(post: Dict) -> str:
    """Extract content from Reddit post."""
    if post.get('selftext'):
        return post.get('selftext', '')[:500]
    elif post.get('url'):
        return f"Link: {post.get('url')}"
    return ""

def get_comment_content(comment: Dict) -> str:
    """Extract content from Reddit comment."""
    return comment.get('body', '')[:500]

def filter_by_date_range(data: List[Dict], start_date: date, end_date: date) -> List[Dict]:
    """Filter data by date range."""
    filtered = []
    
    for item in data:
        created_utc = item.get('created_utc')
        item_date = parse_reddit_timestamp(created_utc)
        
        if item_date and start_date <= item_date <= end_date:
            filtered.append(item)
    
    return filtered

def extract_text_for_keywords(item: Dict) -> str:
    """Extract text content for keyword analysis."""
    text_parts = []
    
    if is_post(item):
        text_parts.append(item.get('title', ''))
        text_parts.append(item.get('selftext', ''))
    else:
        text_parts.append(item.get('body', ''))
    
    return ' '.join([t for t in text_parts if t])


# ------------------ VISUALIZATION DATA GENERATORS ------------------

async def generate_platform_distribution(posts: List[Dict]) -> List[PlatformDistribution]:
    """Generate platform distribution (all Reddit)."""
    total_posts = len(posts)
    return [
        PlatformDistribution(
            platform="Reddit",
            posts_count=total_posts,
            percentage=100.0 if total_posts > 0 else 0.0
        )
    ]

async def generate_engagement_trends(posts: List[Dict], comments: List[Dict]) -> List[EngagementTrend]:
    """Generate monthly engagement trends."""
    monthly_data = defaultdict(lambda: {'reddit': 0})
    
    # Process posts
    for post in posts:
        created_utc = post.get('created_utc')
        post_date = parse_reddit_timestamp(created_utc)
        if post_date:
            month_key = post_date.strftime("%Y-%m")
            monthly_data[month_key]['reddit'] += get_post_engagement(post)
    
    # Process comments
    for comment in comments:
        created_utc = comment.get('created_utc')
        comment_date = parse_reddit_timestamp(created_utc)
        if comment_date:
            month_key = comment_date.strftime("%Y-%m")
            monthly_data[month_key]['reddit'] += get_comment_engagement(comment)
    
    # Convert to response format
    trends = []
    for month, data in sorted(monthly_data.items()):
        reddit_engagement = data['reddit']
        trends.append(EngagementTrend(
            month=month,
            reddit_engagement=reddit_engagement,
            total_engagement=reddit_engagement
        ))
    
    return trends if trends else [EngagementTrend(
        month=datetime.now().strftime("%Y-%m"),
        reddit_engagement=0,
        total_engagement=0
    )]

async def generate_top_posts(posts: List[Dict]) -> List[TopPost]:
    """Generate top 5 posts by engagement."""
    if not posts:
        return []
    
    # Calculate engagement for each post
    post_with_engagement = []
    for post in posts:
        engagement = get_post_engagement(post)
        post_with_engagement.append((post, engagement))
    
    # Sort by engagement and take top 5
    sorted_posts = sorted(post_with_engagement, key=lambda x: x[1], reverse=True)[:5]
    
    top_posts = []
    for rank, (post, engagement) in enumerate(sorted_posts, 1):
        created_utc = post.get('created_utc')
        post_date = parse_reddit_timestamp(created_utc)
        
        top_posts.append(TopPost(
            rank=rank,
            title=post.get('title', 'No Title')[:100],
            content=get_post_content(post),
            engagement_score=engagement,
            platform="Reddit",
            url=get_post_url(post),
            author=post.get('author', 'Unknown'),
            created_date=post_date.isoformat() if post_date else "Unknown"
        ))
    
    return top_posts

async def generate_influential_users(posts: List[Dict], comments: List[Dict]) -> List[InfluentialUser]:
    """Generate top 5 influential users."""
    user_stats = defaultdict(lambda: {'engagement': 0, 'posts': 0, 'comments': 0})
    
    # Process posts
    for post in posts:
        author = post.get('author')
        if author and author not in ['[deleted]', '[removed]']:
            user_stats[author]['engagement'] += get_post_engagement(post)
            user_stats[author]['posts'] += 1
    
    # Process comments
    for comment in comments:
        author = comment.get('author')
        if author and author not in ['[deleted]', '[removed]']:
            user_stats[author]['engagement'] += get_comment_engagement(comment)
            user_stats[author]['comments'] += 1
    
    # Filter users with engagement > 0
    filtered_users = {user: stats for user, stats in user_stats.items() if stats['engagement'] > 0}
    
    # Sort by total engagement
    sorted_users = sorted(filtered_users.items(), key=lambda x: x[1]['engagement'], reverse=True)[:5]
    
    influential_users = []
    for rank, (username, stats) in enumerate(sorted_users, 1):
        total_items = stats['posts'] + stats['comments']
        avg_engagement = stats['engagement'] / total_items if total_items > 0 else 0
        
        influential_users.append(InfluentialUser(
            rank=rank,
            username=username,
            total_engagement=stats['engagement'],
            posts_count=total_items,
            avg_engagement=round(avg_engagement, 2),
            platform="Reddit"
        ))
    
    return influential_users

async def generate_trending_keywords(posts: List[Dict], comments: List[Dict]) -> List[TrendingKeyword]:
    """Generate trending keywords using text analysis."""
    all_text = []
    keyword_posts = defaultdict(set)
    
    # Extract text from posts and comments
    for item in posts + comments:
        text = extract_text_for_keywords(item)
        if text:
            all_text.append(text)
            
            # Simple keyword extraction for post counting
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            for word in words:
                keyword_posts[word].add(item.get('_id', 'unknown'))
    
    if not all_text:
        return []
    
    combined_text = ' '.join(all_text[:2000])  # Limit text for processing
    
    # Try using Groq API for better keyword extraction
    keywords = []
    if client and len(combined_text) > 100:
        try:
            prompt = f"""Extract 10 trending topics or keywords from this Reddit content. 
            Focus on specific nouns, technologies, political topics, or current events. 
            Return ONLY a comma-separated list of keywords.
            
            Content: {combined_text[:3000]}"""
            
            response = await asyncio.to_thread(
                client.chat.completions.create,
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                max_tokens=100,
                temperature=0.3
            )
            
            keywords_text = response.choices[0].message.content.strip()
            extracted_keywords = [kw.strip().lower() for kw in keywords_text.split(',') if kw.strip()]
            
            for keyword in extracted_keywords[:10]:
                freq = combined_text.lower().count(keyword)
                if freq > 1:
                    posts_count = len(keyword_posts.get(keyword, set()))
                    keywords.append(TrendingKeyword(
                        keyword=keyword.title(),
                        frequency=freq,
                        posts_count=posts_count
                    ))
                    
            if keywords:
                return keywords[:10]
                
        except Exception as e:
            logger.error(f"Groq API error: {e}")
    
    # Fallback: simple keyword extraction
    stop_words = {'this', 'that', 'with', 'from', 'have', 'were', 'your', 'they', 
                  'their', 'what', 'which', 'about', 'just', 'like', 'would', 
                  'could', 'https', 'http', 'com', 'www', 'reddit'}
    
    words = re.findall(r'\b[a-zA-Z]{4,}\b', combined_text.lower())
    word_counts = Counter(w for w in words if w not in stop_words)
    
    trending = []
    for word, count in word_counts.most_common(20):
        if len(trending) >= 10:
            break
        posts_count = len(keyword_posts.get(word, set()))
        trending.append(TrendingKeyword(
            keyword=word.title(),
            frequency=count,
            posts_count=posts_count
        ))
    
    return trending

async def generate_posts_histogram(posts: List[Dict]) -> List[PostsHistogram]:
    """Generate weekly posts histogram."""
    weekly_data = defaultdict(lambda: {'reddit': 0})
    
    for post in posts:
        created_utc = post.get('created_utc')
        post_date = parse_reddit_timestamp(created_utc)
        if post_date:
            # Get start of week (Monday)
            week_start = post_date - timedelta(days=post_date.weekday())
            week_key = week_start.strftime("%Y-%m-%d")
            weekly_data[week_key]['reddit'] += 1
    
    histogram = []
    for week, data in sorted(weekly_data.items()):
        reddit_posts = data['reddit']
        histogram.append(PostsHistogram(
            time_period=week,
            reddit_posts=reddit_posts,
            total_posts=reddit_posts
        ))
    
    return histogram if histogram else [PostsHistogram(
        time_period=datetime.now().date().strftime("%Y-%m-%d"),
        reddit_posts=0,
        total_posts=0
    )]

async def generate_trending_comments(comments: List[Dict], posts: List[Dict]) -> List[TrendingComment]:
    """Generate top 5 trending comments."""
    if not comments:
        return []
    
    # Create post title mapping
    post_titles = {}
    for post in posts:
        post_id = post.get('_id')
        if post_id:
            post_titles[post_id] = post.get('title', 'Original Post')[:100]
    
    # Calculate engagement for each comment
    comment_with_engagement = []
    for comment in comments:
        engagement = get_comment_engagement(comment)
        comment_with_engagement.append((comment, engagement))
    
    # Sort by engagement and take top 5
    sorted_comments = sorted(comment_with_engagement, key=lambda x: x[1], reverse=True)[:5]
    
    trending_comments = []
    for rank, (comment, engagement) in enumerate(sorted_comments, 1):
        link_id = comment.get('link_id', '')
        if link_id.startswith('t3_'):
            post_id = link_id[3:]
        else:
            post_id = link_id
        
        post_title = post_titles.get(post_id, "Original Post")
        
        trending_comments.append(TrendingComment(
            rank=rank,
            author=comment.get('author', 'Unknown'),
            content=get_comment_content(comment),
            engagement_score=engagement,
            platform="Reddit",
            url=get_comment_url(comment),
            post_title=post_title
        ))
    
    return trending_comments

async def generate_sentiment_trends(posts: List[Dict], comments: List[Dict]) -> SentimentTrend:
    """Generate sentiment trends (simplified version)."""
    # Since we don't have actual sentiment data, we'll generate mock data
    # based on engagement scores
    monthly_data = defaultdict(lambda: {'pos': 0, 'neg': 0, 'neu': 0, 'count': 0})
    
    # Process posts
    for post in posts:
        created_utc = post.get('created_utc')
        post_date = parse_reddit_timestamp(created_utc)
        if post_date:
            month_key = post_date.strftime("%Y-%m")
            engagement = get_post_engagement(post)
            
            # Simplified sentiment based on engagement
            if engagement > 20:
                monthly_data[month_key]['pos'] += 0.7
                monthly_data[month_key]['neu'] += 0.2
                monthly_data[month_key]['neg'] += 0.1
            elif engagement > 5:
                monthly_data[month_key]['pos'] += 0.4
                monthly_data[month_key]['neu'] += 0.4
                monthly_data[month_key]['neg'] += 0.2
            else:
                monthly_data[month_key]['pos'] += 0.2
                monthly_data[month_key]['neu'] += 0.5
                monthly_data[month_key]['neg'] += 0.3
            
            monthly_data[month_key]['count'] += 1
    
    # Process comments similarly
    for comment in comments:
        created_utc = comment.get('created_utc')
        comment_date = parse_reddit_timestamp(created_utc)
        if comment_date:
            month_key = comment_date.strftime("%Y-%m")
            engagement = get_comment_engagement(comment)
            
            if engagement > 10:
                monthly_data[month_key]['pos'] += 0.6
                monthly_data[month_key]['neu'] += 0.3
                monthly_data[month_key]['neg'] += 0.1
            elif engagement > 2:
                monthly_data[month_key]['pos'] += 0.3
                monthly_data[month_key]['neu'] += 0.5
                monthly_data[month_key]['neg'] += 0.2
            else:
                monthly_data[month_key]['pos'] += 0.1
                monthly_data[month_key]['neu'] += 0.6
                monthly_data[month_key]['neg'] += 0.3
            
            monthly_data[month_key]['count'] += 1
    
    # Calculate averages
    sentiment_points = []
    for month, data in sorted(monthly_data.items()):
        if data['count'] > 0:
            sentiment_points.append(SentimentPoint(
                month=month,
                positive_score=round(data['pos'] / data['count'], 3),
                negative_score=round(data['neg'] / data['count'], 3),
                neutral_score=round(data['neu'] / data['count'], 3)
            ))
    
    return SentimentTrend(reddit=sentiment_points if sentiment_points else [])


# ------------------ API ROUTES ------------------

@app.get("/")
async def root():
    return {
        "message": "Reddit Insights API v5.0",
        "status": "active",
        "endpoints": [
            "/dashboard - Complete dashboard data",
            "/platform-distribution - Platform stats",
            "/engagement-trends - Monthly engagement",
            "/top-posts - Top 5 posts",
            "/influential-users - Top 5 users",
            "/trending-keywords - Trending topics",
            "/posts-histogram - Weekly post count",
            "/trending-comments - Top 5 comments",
            "/sentiment-trends - Monthly sentiment"
        ]
    }

@app.get("/health")
async def health_check():
    data = load_reddit_data()
    return {
        "status": "healthy",
        "groq_available": client is not None,
        "reddit_data_loaded": len(data) > 0,
        "total_records": len(data),
        "posts_count": len([d for d in data if is_post(d)]),
        "comments_count": len([d for d in data if is_comment(d)])
    }

@app.get("/debug/data-stats")
async def debug_data_stats():
    """Debug endpoint to understand data structure."""
    data = load_reddit_data()
    
    if not data:
        return {"error": "No data loaded"}
    
    # Sample a few records
    samples = []
    post_count = 0
    comment_count = 0
    
    for item in data[:5]:
        samples.append({
            "kind": item.get('_kind'),
            "id": item.get('_id'),
            "author": item.get('author'),
            "score": item.get('score'),
            "title": item.get('title', '')[:50] if item.get('title') else None,
            "has_selftext": 'selftext' in item,
            "created_utc": item.get('created_utc'),
            "num_comments": item.get('num_comments'),
            "is_post": is_post(item),
            "is_comment": is_comment(item)
        })
        
        if is_post(item):
            post_count += 1
        elif is_comment(item):
            comment_count += 1
    
    return {
        "total_records": len(data),
        "posts": post_count,
        "comments": comment_count,
        "samples": samples
    }

@app.post("/dashboard", response_model=DashboardResponse)
async def get_complete_dashboard(request: PlatformRequest):
    """Main dashboard endpoint with all visualizations."""
    try:
        start_time = datetime.now()
        logger.info(f"Generating dashboard from {request.date_range.start_date} to {request.date_range.end_date}")
        
        # Load and filter data
        all_data = load_reddit_data()
        filtered_data = filter_by_date_range(all_data, request.date_range.start_date, request.date_range.end_date)
        
        # Separate posts and comments
        posts = [item for item in filtered_data if is_post(item)]
        comments = [item for item in filtered_data if is_comment(item)]
        
        logger.info(f"Processing {len(posts)} posts and {len(comments)} comments")
        
        # Calculate totals
        total_posts = len(posts)
        total_engagement = sum(get_post_engagement(post) for post in posts) + sum(get_comment_engagement(comment) for comment in comments)
        
        # Generate all visualizations concurrently
        tasks = [
            generate_platform_distribution(posts),
            generate_engagement_trends(posts, comments),
            generate_top_posts(posts),
            generate_influential_users(posts, comments),
            generate_trending_keywords(posts, comments),
            generate_posts_histogram(posts),
            generate_trending_comments(comments, posts),
            generate_sentiment_trends(posts, comments)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in visualization {i}: {result}")
                processed_results.append([])  # Return empty list for failed viz
            else:
                processed_results.append(result)
        
        # Prepare response
        response = DashboardResponse(
            total_posts=total_posts,
            total_engagement=total_engagement,
            influential_users_count=len(processed_results[3]),
            trending_keywords_count=len(processed_results[4]),
            platform_distribution=processed_results[0],
            engagement_trends=processed_results[1],
            top_posts=processed_results[2],
            influential_users=processed_results[3],
            trending_keywords=processed_results[4],
            posts_histogram=processed_results[5],
            trending_comments=processed_results[6],
            sentiment_trends=processed_results[7]
        )
        
        logger.info(f"Dashboard generated in {(datetime.now() - start_time).total_seconds():.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Individual endpoints for each visualization
@app.post("/platform-distribution")
async def get_platform_distribution(request: PlatformRequest):
    """Get platform distribution data."""
    data = load_reddit_data()
    filtered = filter_by_date_range(data, request.date_range.start_date, request.date_range.end_date)
    posts = [item for item in filtered if is_post(item)]
    return await generate_platform_distribution(posts)

@app.post("/engagement-trends")
async def get_engagement_trends(request: PlatformRequest):
    """Get engagement trends data."""
    data = load_reddit_data()
    filtered = filter_by_date_range(data, request.date_range.start_date, request.date_range.end_date)
    posts = [item for item in filtered if is_post(item)]
    comments = [item for item in filtered if is_comment(item)]
    return await generate_engagement_trends(posts, comments)

@app.post("/top-posts")
async def get_top_posts(request: PlatformRequest):
    """Get top posts data."""
    data = load_reddit_data()
    filtered = filter_by_date_range(data, request.date_range.start_date, request.date_range.end_date)
    posts = [item for item in filtered if is_post(item)]
    return await generate_top_posts(posts)

@app.post("/influential-users")
async def get_influential_users(request: PlatformRequest):
    """Get influential users data."""
    data = load_reddit_data()
    filtered = filter_by_date_range(data, request.date_range.start_date, request.date_range.end_date)
    posts = [item for item in filtered if is_post(item)]
    comments = [item for item in filtered if is_comment(item)]
    return await generate_influential_users(posts, comments)

@app.post("/trending-keywords")
async def get_trending_keywords(request: PlatformRequest):
    """Get trending keywords data."""
    data = load_reddit_data()
    filtered = filter_by_date_range(data, request.date_range.start_date, request.date_range.end_date)
    posts = [item for item in filtered if is_post(item)]
    comments = [item for item in filtered if is_comment(item)]
    return await generate_trending_keywords(posts, comments)

@app.post("/posts-histogram")
async def get_posts_histogram(request: PlatformRequest):
    """Get posts histogram data."""
    data = load_reddit_data()
    filtered = filter_by_date_range(data, request.date_range.start_date, request.date_range.end_date)
    posts = [item for item in filtered if is_post(item)]
    return await generate_posts_histogram(posts)

@app.post("/trending-comments")
async def get_trending_comments(request: PlatformRequest):
    """Get trending comments data."""
    data = load_reddit_data()
    filtered = filter_by_date_range(data, request.date_range.start_date, request.date_range.end_date)
    posts = [item for item in filtered if is_post(item)]
    comments = [item for item in filtered if is_comment(item)]
    return await generate_trending_comments(comments, posts)

@app.post("/sentiment-trends")
async def get_sentiment_trends(request: PlatformRequest):
    """Get sentiment trends data."""
    data = load_reddit_data()
    filtered = filter_by_date_range(data, request.date_range.start_date, request.date_range.end_date)
    posts = [item for item in filtered if is_post(item)]
    comments = [item for item in filtered if is_comment(item)]
    return await generate_sentiment_trends(posts, comments)

@app.get("/available-dates")
async def get_available_dates():
    """Get min and max dates in the dataset."""
    data = load_reddit_data()
    dates = []
    
    for item in data:
        created_utc = item.get('created_utc')
        item_date = parse_reddit_timestamp(created_utc)
        if item_date:
            dates.append(item_date)
    
    if dates:
        return {
            "min_date": min(dates).isoformat(),
            "max_date": max(dates).isoformat()
        }
    else:
        return {
            "min_date": None,
            "max_date": None
        }

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting Reddit Insights API v5.0")
    data = load_reddit_data()
    logger.info(f"Loaded {len(data)} Reddit records")
    logger.info(f"Groq client available: {client is not None}")
    logger.info("API is ready.")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API")
    executor.shutdown(wait=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")