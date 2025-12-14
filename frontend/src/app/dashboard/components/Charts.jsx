'use client';

import React, { useMemo } from 'react';
import { Bar, Line } from 'react-chartjs-2';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from '@/lib/utils';

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { 
      position: 'top', 
      labels: { 
        padding: 20, 
        usePointStyle: true, 
        font: { size: 13, weight: '600' },
        color: '#000'
      } 
    },
    tooltip: {
      backgroundColor: 'rgba(0, 0, 0, 0.95)',
      titleColor: '#fff',
      bodyColor: '#fff',
      borderColor: '#000',
      borderWidth: 2,
      padding: 12,
      callbacks: {
        label: function(context) {
          let label = context.dataset.label || '';
          if (label) label += ': ';
          let value = context.parsed.y ?? context.parsed;
          if (value !== null) {
            return label + new Intl.NumberFormat('en-US').format(value);
          }
          return label;
        }
      }
    }
  },
  scales: {
    x: { 
      grid: { display: false },
      ticks: {
        maxRotation: 45,
        minRotation: 45,
        color: '#000',
        font: { size: 11, weight: '600' }
      },
      border: { color: '#000', width: 2 }
    },
    y: {
      grid: { color: 'rgba(0, 0, 0, 0.1)', lineWidth: 1 },
      beginAtZero: true,
      ticks: {
        color: '#000',
        font: { size: 11, weight: '600' },
        callback: function(value) {
          if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
          if (value >= 1000) return (value / 1000).toFixed(1) + 'K';
          return value;
        }
      },
      border: { color: '#000', width: 2 }
    }
  }
};

export const PostsHistogramChart = ({ data, className = "" }) => {
  const chartData = useMemo(() => {
    if (!data || !data.posts_histogram || data.posts_histogram.length === 0) {
      return null;
    }

    const histogram = data.posts_histogram.slice(-12);
    
    return {
      labels: histogram.map(item => {
        const date = new Date(item.time_period);
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      }),
      datasets: [
        {
          label: 'Reddit Posts',
          data: histogram.map(item => item.reddit_posts || item.total_posts || 0),
          backgroundColor: 'rgba(0, 0, 0, 0.9)',
          borderColor: 'rgb(0, 0, 0)',
          borderWidth: 2,
          hoverBackgroundColor: 'rgba(60, 60, 60, 0.9)',
        }
      ]
    };
  }, [data]);

  if (!chartData) {
    return (
      <Card className={cn("border-2 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]", className)}>
        <CardHeader className="pb-4 border-b-2 border-black">
          <CardTitle className="text-xl font-bold tracking-tight">Weekly Posts Distribution</CardTitle>
          <CardDescription className="text-gray-600 mt-1">Number of posts per week</CardDescription>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="h-72 flex items-center justify-center text-gray-400">
            No post histogram data available
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn("border-2 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] transition-all duration-300", className)}>
      <CardHeader className="pb-4 border-b-2 border-black">
        <CardTitle className="text-xl font-bold tracking-tight">Weekly Posts Distribution</CardTitle>
        <CardDescription className="text-gray-600 mt-1">Number of posts per week</CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="h-72 relative">
          <Bar options={chartOptions} data={chartData} />
        </div>
      </CardContent>
    </Card>
  );
};

export const EngagementTrendsChart = ({ data, className = "" }) => {
  const chartData = useMemo(() => {
    if (!data || !data.engagement_trends || data.engagement_trends.length === 0) {
      return null;
    }

    const trends = data.engagement_trends;
    
    return {
      labels: trends.map(item => {
        const [year, month] = item.month.split('-');
        const date = new Date(year, month - 1);
        return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
      }),
      datasets: [
        {
          label: 'Reddit Engagement',
          data: trends.map(item => item.reddit_engagement || 0),
          borderColor: 'rgb(0, 0, 0)',
          backgroundColor: 'rgba(0, 0, 0, 0.1)',
          borderWidth: 3,
          tension: 0.4,
          fill: true,
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: '#fff',
          pointBorderColor: '#000',
          pointBorderWidth: 2,
        }
      ]
    };
  }, [data]);

  if (!chartData) {
    return (
      <Card className={cn("border-2 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]", className)}>
        <CardHeader className="pb-4 border-b-2 border-black">
          <CardTitle className="text-xl font-bold tracking-tight">Engagement Trends</CardTitle>
          <CardDescription className="text-gray-600 mt-1">Monthly engagement scores</CardDescription>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="h-72 flex items-center justify-center text-gray-400">
            No engagement trends data available
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn("border-2 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] transition-all duration-300", className)}>
      <CardHeader className="pb-4 border-b-2 border-black">
        <CardTitle className="text-xl font-bold tracking-tight">Engagement Trends</CardTitle>
        <CardDescription className="text-gray-600 mt-1">Monthly engagement scores</CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="h-72 relative">
          <Line options={chartOptions} data={chartData} />
        </div>
      </CardContent>
    </Card>
  );
};

export const SentimentTrendsChart = ({ data, className = "" }) => {
  const chartData = useMemo(() => {
    if (!data || !data.sentiment_trends || !data.sentiment_trends.reddit || 
        data.sentiment_trends.reddit.length === 0) {
      return null;
    }

    const sentimentData = data.sentiment_trends.reddit;
    
    return {
      labels: sentimentData.map(item => {
        const [year, month] = item.month.split('-');
        const date = new Date(year, month - 1);
        return date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
      }),
      datasets: [
        {
          label: 'Positive',
          data: sentimentData.map(item => item.positive_score || 0),
          borderColor: 'rgba(30, 30, 30, 1)',
          backgroundColor: 'rgba(30, 30, 30, 0.1)',
          borderWidth: 3,
          tension: 0.4,
          fill: true,
          pointRadius: 4,
          pointBackgroundColor: '#fff',
          pointBorderColor: 'rgba(30, 30, 30, 1)',
          pointBorderWidth: 2,
        },
        {
          label: 'Neutral',
          data: sentimentData.map(item => item.neutral_score || 0),
          borderColor: 'rgba(90, 90, 90, 1)',
          backgroundColor: 'rgba(90, 90, 90, 0.1)',
          borderWidth: 3,
          tension: 0.4,
          fill: true,
          pointRadius: 4,
          pointBackgroundColor: '#fff',
          pointBorderColor: 'rgba(90, 90, 90, 1)',
          pointBorderWidth: 2,
        },
        {
          label: 'Negative',
          data: sentimentData.map(item => item.negative_score || 0),
          borderColor: 'rgba(150, 150, 150, 1)',
          backgroundColor: 'rgba(150, 150, 150, 0.1)',
          borderWidth: 3,
          tension: 0.4,
          fill: true,
          pointRadius: 4,
          pointBackgroundColor: '#fff',
          pointBorderColor: 'rgba(150, 150, 150, 1)',
          pointBorderWidth: 2,
        }
      ]
    };
  }, [data]);

  if (!chartData) {
    return (
      <Card className={cn("border-2 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]", className)}>
        <CardHeader className="pb-4 border-b-2 border-black">
          <CardTitle className="text-xl font-bold tracking-tight">Sentiment Trends</CardTitle>
          <CardDescription className="text-gray-600 mt-1">Monthly sentiment analysis</CardDescription>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="h-72 flex items-center justify-center text-gray-400">
            No sentiment trends data available
          </div>
        </CardContent>
      </Card>
    );
  }

  const sentimentOptions = {
    ...chartOptions,
    scales: {
      ...chartOptions.scales,
      y: {
        ...chartOptions.scales.y,
        min: 0,
        max: 1,
        ticks: {
          color: '#000',
          font: { size: 11, weight: '600' },
          callback: function(value) {
            return (value * 100).toFixed(0) + '%';
          }
        }
      }
    }
  };

  return (
    <Card className={cn("border-2 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] transition-all duration-300", className)}>
      <CardHeader className="pb-4 border-b-2 border-black">
        <CardTitle className="text-xl font-bold tracking-tight">Sentiment Trends</CardTitle>
        <CardDescription className="text-gray-600 mt-1">Monthly sentiment scores (0-1 scale)</CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="h-72 relative">
          <Line options={sentimentOptions} data={chartData} />
        </div>
      </CardContent>
    </Card>
  );
};