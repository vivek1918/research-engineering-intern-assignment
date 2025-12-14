'use client';

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ExternalLink, Users, Hash } from 'lucide-react';

export const PostsList = ({ data }) => {
  if (!data || !data.top_posts || data.top_posts.length === 0) {
    return (
      <Card className="border-2 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
        <CardHeader className="border-b-2 border-black">
          <CardTitle className="text-lg font-bold flex items-center gap-2">
            <ExternalLink className="h-5 w-5" />
            Top Posts
          </CardTitle>
          <CardDescription className="text-gray-600">Most engaging posts</CardDescription>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="text-center text-gray-400 py-8">
            No top posts data available
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-2 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] transition-all duration-300">
      <CardHeader className="border-b-2 border-black">
        <CardTitle className="text-lg font-bold flex items-center gap-2">
          <ExternalLink className="h-5 w-5" />
          Top Posts
        </CardTitle>
        <CardDescription className="text-gray-600">Most engaging posts</CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-4">
          {data.top_posts.slice(0, 5).map((post) => (
            <div key={post.rank} className="border-2 border-black rounded-lg p-4 hover:bg-gray-50 transition-all duration-200 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[6px_6px_0px_0px_rgba(0,0,0,1)]">
              <div className="flex justify-between items-start mb-3">
                <div className="flex-1">
                  <Badge variant="outline" className="mb-2 border-2 border-black font-bold">
                    #{post.rank} • {post.platform}
                  </Badge>
                  <h4 className="font-bold line-clamp-2 text-black">{post.title}</h4>
                  <p className="text-sm text-gray-600 mt-2 line-clamp-2">
                    {post.content}
                  </p>
                </div>
                <div className="text-right ml-4">
                  <div className="font-black text-2xl">{post.engagement_score}</div>
                  <div className="text-xs text-gray-600 font-semibold uppercase">Engagement</div>
                </div>
              </div>
              <div className="flex justify-between items-center mt-3 pt-3 border-t-2 border-black">
                <span className="text-sm text-gray-600 font-semibold">by {post.author}</span>
                {post.url !== '#' && (
                  <a 
                    href={post.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-sm text-black font-bold hover:underline flex items-center gap-1"
                  >
                    View <ExternalLink className="h-3 w-3" />
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export const UsersList = ({ data }) => {
  if (!data || !data.influential_users || data.influential_users.length === 0) {
    return (
      <Card className="border-2 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
        <CardHeader className="border-b-2 border-black">
          <CardTitle className="text-lg font-bold flex items-center gap-2">
            <Users className="h-5 w-5" />
            Influential Users
          </CardTitle>
          <CardDescription className="text-gray-600">Most active users</CardDescription>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="text-center text-gray-400 py-8">
            No influential users data available
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-2 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] transition-all duration-300">
      <CardHeader className="border-b-2 border-black">
        <CardTitle className="text-lg font-bold flex items-center gap-2">
          <Users className="h-5 w-5" />
          Influential Users
        </CardTitle>
        <CardDescription className="text-gray-600">Most active users</CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-4">
          {data.influential_users.slice(0, 5).map((user) => (
            <div key={user.rank} className="border-2 border-black rounded-lg p-4 hover:bg-gray-50 transition-all duration-200 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[6px_6px_0px_0px_rgba(0,0,0,1)]">
              <div className="flex justify-between items-center mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full bg-black flex items-center justify-center border-2 border-black">
                    <Users className="h-6 w-6 text-white" />
                  </div>
                  <div>
                    <h4 className="font-bold text-black">{user.username}</h4>
                    <p className="text-sm text-gray-600 font-semibold">{user.platform}</p>
                  </div>
                </div>
                <Badge variant="secondary" className="border-2 border-black font-bold">#{user.rank}</Badge>
              </div>
              <div className="grid grid-cols-3 gap-4 pt-4 border-t-2 border-black">
                <div className="text-center">
                  <div className="font-black text-lg">{user.total_engagement.toLocaleString()}</div>
                  <div className="text-xs text-gray-600 font-semibold uppercase">Engagement</div>
                </div>
                <div className="text-center">
                  <div className="font-black text-lg">{user.posts_count}</div>
                  <div className="text-xs text-gray-600 font-semibold uppercase">Posts</div>
                </div>
                <div className="text-center">
                  <div className="font-black text-lg">{user.avg_engagement.toFixed(1)}</div>
                  <div className="text-xs text-gray-600 font-semibold uppercase">Avg/Post</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export const KeywordsList = ({ data }) => {
  if (!data || !data.trending_keywords || data.trending_keywords.length === 0) {
    return (
      <Card className="border-2 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
        <CardHeader className="border-b-2 border-black">
          <CardTitle className="text-lg font-bold flex items-center gap-2">
            <Hash className="h-5 w-5" />
            Trending Keywords
          </CardTitle>
          <CardDescription className="text-gray-600">Most mentioned topics</CardDescription>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="text-center text-gray-400 py-8">
            No trending keywords data available
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-2 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] transition-all duration-300 lg:col-span-2">
      <CardHeader className="border-b-2 border-black">
        <CardTitle className="text-lg font-bold flex items-center gap-2">
          <Hash className="h-5 w-5" />
          Trending Keywords
        </CardTitle>
        <CardDescription className="text-gray-600">Most mentioned topics</CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {data.trending_keywords.slice(0, 10).map((keyword, index) => (
            <div key={index} className="border-2 border-black rounded-lg p-4 hover:bg-gray-50 transition-all duration-200 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[6px_6px_0px_0px_rgba(0,0,0,1)]">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-3 flex-1">
                  <div className="w-10 h-10 rounded-full bg-black flex items-center justify-center border-2 border-black flex-shrink-0">
                    <Hash className="h-5 w-5 text-white" />
                  </div>
                  <div className="min-w-0">
                    <h4 className="font-bold text-black truncate">{keyword.keyword}</h4>
                    <p className="text-sm text-gray-600 font-semibold">
                      {keyword.posts_count} posts
                    </p>
                  </div>
                </div>
                <div className="text-right ml-4 flex-shrink-0">
                  <div className="font-black text-xl">{keyword.frequency}</div>
                  <div className="text-xs text-gray-600 font-semibold uppercase">Mentions</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};