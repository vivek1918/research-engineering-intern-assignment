import { NextResponse } from 'next/server';

// Simple in-memory cache
const cache = new Map();

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  
  const cacheKey = searchParams.toString();
  const CACHE_DURATION_MS = 10 * 60 * 1000; // 10 minutes

  // 1. Check if a valid, non-expired result is in the cache
  if (cache.has(cacheKey)) {
    const cachedEntry = cache.get(cacheKey);
    if (Date.now() - cachedEntry.timestamp < CACHE_DURATION_MS) {
      console.log(`✅ Returning cached data for: ${cacheKey}`);
      return NextResponse.json(cachedEntry.data);
    }
  }
  
  console.log(`🚀 Fetching new data for: ${cacheKey}`);
  
  const query = searchParams.get('q');
  const category = searchParams.get('category');
  const page = searchParams.get('page') || '1';
  // ADDED: Read the 'from' and 'to' date parameters from the request URL
  const fromDate = searchParams.get('from');
  const toDate = searchParams.get('to');
  
  const apiKey = process.env.NEWS_API_KEY;

  if (!apiKey) {
    console.error("API Route Error: NEWS_API_KEY environment variable is not set.");
    return NextResponse.json(
      { error: 'Server configuration error: News API key is missing.' },
      { status: 500 }
    );
  }

  const pageSize = 20;
  let apiUrl = 'https://newsapi.org/v2/';

  if (query) {
    apiUrl += `everything?q=${encodeURIComponent(query)}`;
  } else {
    apiUrl += `top-headlines?country=us`;
    if (category && category !== 'general') {
      apiUrl += `&category=${category}`;
    }
  }

  // ADDED: Append the date parameters to the API URL if they exist.
  // The NewsAPI will automatically apply these filters if the endpoint is /everything.
  // It will safely ignore them if the endpoint is /top-headlines.
  if (fromDate && toDate) {
    apiUrl += `&from=${fromDate}&to=${toDate}`;
  }

  apiUrl += `&pageSize=${pageSize}&page=${page}&apiKey=${apiKey}`;

  try {
    const res = await fetch(apiUrl);
    const data = await res.json();

    if (data.status === 'error') {
      console.error(`NewsAPI Error: ${data.message}`);
      return NextResponse.json({ error: data.message }, { status: 400 });
    }
    
    // 2. If the fetch is successful, store the new data in the cache
    const cacheEntry = {
      timestamp: Date.now(),
      data: data,
    };
    cache.set(cacheKey, cacheEntry);

    return NextResponse.json(data);
    
  } catch (error) {
    console.error("Failed to fetch from NewsAPI:", error);
    return NextResponse.json(
      { error: 'An external API error occurred.' },
      { status: 502 }
    );
  }
}