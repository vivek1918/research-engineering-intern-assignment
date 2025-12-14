'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { NewsCard } from '../components/NewsCard';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Search, Newspaper, AlertCircle, Sparkles } from 'lucide-react';

const staticDomains = [
  { name: 'YouTube', query: 'youtube.com' },
  { name: 'Reddit', query: 'reddit.com' },
];
const categories = ['Business', 'Entertainment', 'Sports', 'Technology'];

const SkeletonGrid = () => (
  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
    {Array.from({ length: 12 }).map((_, index) => (
      <div key={index} className="bg-white p-4 rounded-lg border-2 border-black shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] animate-pulse space-y-3">
        <div className="bg-gray-200 h-40 rounded-md"></div>
        <div className="bg-gray-300 h-5 rounded w-3/4"></div>
        <div className="bg-gray-200 h-4 rounded w-full"></div>
        <div className="bg-gray-200 h-4 rounded w-5/6"></div>
        <div className="flex justify-between pt-2">
          <div className="bg-gray-300 h-4 rounded w-1/4"></div>
          <div className="bg-gray-300 h-4 rounded w-1/4"></div>
        </div>
      </div>
    ))}
  </div>
);

export default function NewsPage() {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [activeFilter, setActiveFilter] = useState('general');

  const fetchNews = useCallback(async (filter, search) => {
    setLoading(true);
    setArticles([]);
    setError(null);
    
    let url = `/api/news?page=1`; 
    
    const domain = staticDomains.find((d) => d.name.toLowerCase() === filter);
    const currentSearchTerm = search.trim();

    if (currentSearchTerm) {
      url += `&q=${encodeURIComponent(currentSearchTerm)}`;
    } else if (domain) {
      url += `&q=${encodeURIComponent(domain.query)}`;
    } else if (filter !== 'general') {
      url += `&category=${filter}`;
    }

    try {
      const fetchPromise = fetch(url).then((res) => {
        if (!res.ok) throw new Error('Network response was not ok');
        return res.json();
      });
      const delayPromise = new Promise((resolve) => setTimeout(resolve, 500));
      const [data] = await Promise.all([fetchPromise, delayPromise]);
      if (data.error) throw new Error(data.error);
      
      if (data.articles && data.articles.length > 0) {
        setArticles(data.articles);
      } else {
        setArticles([]);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);
  
  useEffect(() => {
    fetchNews(activeFilter, searchTerm);
  }, [activeFilter, searchTerm, fetchNews]);

  const handleSearch = (e) => {
    e.preventDefault();
    const newSearchTerm = e.target.elements.search.value;
    setActiveFilter('');
    setSearchTerm(newSearchTerm);
  };

  const handleFilterClick = (newFilter) => {
    const filter = newFilter.toLowerCase();
    setSearchTerm('');
    if (document.querySelector('input[name="search"]')) {
      document.querySelector('input[name="search"]').value = '';
    }
    setActiveFilter(filter);
  };

  const allFilters = ['General', ...staticDomains.map(d => d.name), ...categories];

  return (
    <div className="space-y-8 p-6 bg-white min-h-screen">
      <header className="border-b-4 border-black pb-6">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-12 h-12 rounded-full bg-black flex items-center justify-center border-2 border-black shadow-[6px_6px_0px_0px_rgba(0,0,0,1)]">
            <Newspaper className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-black tracking-tighter">News Feed</h1>
            <p className="text-gray-600 text-lg font-semibold">Your real-time window into global headlines</p>
          </div>
        </div>
      </header>

      <div className="space-y-6">
        <form onSubmit={handleSearch} className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-600" />
          <Input 
            name="search" 
            type="text" 
            placeholder="Search for any topic..." 
            className="pl-12 pr-4 py-3 border-2 border-black rounded-lg shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] focus:shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] transition-all font-semibold text-base focus:outline-none focus:ring-2 focus:ring-black"
          />
        </form>

        <div className="flex flex-wrap gap-3">
          {allFilters.map((filterName) => (
            <Button
              key={filterName}
              variant={activeFilter === filterName.toLowerCase() ? 'default' : 'outline'}
              onClick={() => handleFilterClick(filterName)}
              className={`border-2 border-black font-bold shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] transition-all ${
                activeFilter === filterName.toLowerCase() 
                  ? 'bg-black text-white hover:bg-gray-800' 
                  : 'bg-white text-black hover:bg-gray-100'
              }`}
            >
              {filterName}
            </Button>
          ))}
        </div>
      </div>
      
      {loading && <SkeletonGrid />}
      
      {error && (
        <div className="bg-white border-2 border-black rounded-lg p-6 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
          <div className="flex items-center gap-3 text-black">
            <AlertCircle className="h-6 w-6 flex-shrink-0" />
            <div>
              <h3 className="font-bold text-lg mb-1">Error Loading News</h3>
              <p className="font-semibold text-gray-700">{error}</p>
            </div>
          </div>
        </div>
      )}
      
      {!loading && articles.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {articles.slice(0, 12).map((article) => (
            <NewsCard key={article.url} article={article} />
          ))}
        </div>
      )}
      
      {!loading && !error && articles.length === 0 && (
        <div className="bg-white border-2 border-black rounded-lg p-12 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] text-center">
          <div className="w-16 h-16 rounded-full bg-gray-100 border-2 border-black flex items-center justify-center mx-auto mb-4">
            <Sparkles className="h-8 w-8 text-black" />
          </div>
          <h3 className="text-2xl font-black mb-2">No Articles Found</h3>
          <p className="text-gray-600 font-semibold">Try a different search term or category to discover more news.</p>
        </div>
      )}
    </div>
  );
}