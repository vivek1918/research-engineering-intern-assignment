'use client';

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ExternalLink, Calendar, User } from 'lucide-react';
import { cn } from '@/lib/utils';

export const NewsCard = ({ article }) => {
  const { title, description, url, urlToImage, publishedAt, source, author } = article;
  
  const formattedDate = publishedAt 
    ? new Date(publishedAt).toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric', 
        year: 'numeric' 
      })
    : 'Unknown date';

  const handleClick = () => {
    if (url) {
      window.open(url, '_blank', 'noopener,noreferrer');
    }
  };

  return (
    <Card 
      className="group cursor-pointer border-2 border-black shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] hover:shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] transition-all duration-300 hover:-translate-y-1 overflow-hidden bg-white"
      onClick={handleClick}
    >
      {urlToImage && (
        <div className="relative h-48 overflow-hidden border-b-2 border-black">
          <img
            src={urlToImage}
            alt={title || 'News article'}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
            onError={(e) => {
              e.target.style.display = 'none';
              e.target.parentElement.style.display = 'none';
            }}
          />
          {source?.name && (
            <div className="absolute top-3 right-3">
              <Badge 
                variant="secondary" 
                className="bg-white border-2 border-black font-bold shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]"
              >
                {source.name}
              </Badge>
            </div>
          )}
        </div>
      )}
      
      <CardHeader className="pb-3 border-b-2 border-black">
        <CardTitle className="text-base font-bold line-clamp-2 group-hover:underline text-black">
          {title || 'Untitled Article'}
        </CardTitle>
      </CardHeader>
      
      <CardContent className="pt-4">
        {description && (
          <CardDescription className="text-sm line-clamp-3 mb-4 text-gray-700 font-medium">
            {description}
          </CardDescription>
        )}
        
        <div className="flex flex-col gap-2 text-xs text-gray-600 font-semibold pt-3 border-t-2 border-black">
          <div className="flex items-center gap-2">
            <Calendar className="h-3 w-3" />
            <span>{formattedDate}</span>
          </div>
          
          {author && (
            <div className="flex items-center gap-2">
              <User className="h-3 w-3" />
              <span className="truncate">{author}</span>
            </div>
          )}
          
          <div className="flex items-center gap-2 text-black font-bold mt-2">
            <ExternalLink className="h-3 w-3" />
            <span>Read Article</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};