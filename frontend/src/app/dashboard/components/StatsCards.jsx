'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Rss, Activity, Hash, Users } from 'lucide-react';

export const StatsCardGrid = ({ data }) => {
  const cards = [
    {
      title: "Total Posts",
      value: data?.totalPosts?.toLocaleString() || "0",
      description: "Posts in selected period",
      icon: Rss,
      gradient: "from-gray-900 to-gray-700"
    },
    {
      title: "Total Engagement",
      value: data?.totalEngagement?.toLocaleString() || "0",
      description: "Engagement score",
      icon: Activity,
      gradient: "from-gray-800 to-gray-600"
    },
    {
      title: "Trending Keywords",
      value: data?.totalKeywords?.toLocaleString() || "0",
      description: "Unique trending topics",
      icon: Hash,
      gradient: "from-gray-700 to-gray-500"
    },
    {
      title: "Influential Users",
      value: data?.totalUsers?.toLocaleString() || "0",
      description: "Active contributors",
      icon: Users,
      gradient: "from-gray-600 to-gray-400"
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {cards.map((card, index) => (
        <Card 
          key={index} 
          className="border-2 border-black shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] hover:shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] transition-all duration-300 hover:-translate-y-1"
        >
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 border-b-2 border-black">
            <CardTitle className="text-sm font-bold uppercase tracking-wide">
              {card.title}
            </CardTitle>
            <div className={`p-3 rounded-lg bg-gradient-to-br ${card.gradient} border-2 border-black`}>
              <card.icon className="h-5 w-5 text-white" />
            </div>
          </CardHeader>
          <CardContent className="pt-4">
            <div className="text-3xl font-black tracking-tight">{card.value}</div>
            <p className="text-xs text-gray-600 mt-2 font-medium uppercase tracking-wide">
              {card.description}
            </p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};