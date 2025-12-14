'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { Bar, Line, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement, LineElement,
  PointElement, ArcElement, Title, Tooltip, Legend
} from 'chart.js';
import {
  Calendar as CalendarIcon,
  Rss,
  Users,
  MessageSquare,
  ExternalLink,
  Hash,
  Eye,
  Activity,
  AlertCircle,
  MessageCircle,
  TrendingUp,
  BarChart3
} from 'lucide-react';
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { addDays, format } from 'date-fns';
import { cn } from '@/lib/utils';
import { StatsCardGrid } from './components/StatsCards';
import { PostsHistogramChart, EngagementTrendsChart, SentimentTrendsChart } from './components/Charts';
import { PostsList, UsersList, KeywordsList } from './components/ListCards';

ChartJS.register(
  CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement, Title, Tooltip, Legend
);

const DASHBOARD_API = process.env.NEXT_PUBLIC_DASHBOARD_API || "http://localhost:8000";

// Modern Black & White Color Palettes
const COLORS = {
  reddit: {
    solid: 'rgb(0, 0, 0)',
    light: 'rgba(0, 0, 0, 0.8)',
    faded: 'rgba(0, 0, 0, 0.05)',
    border: 'rgba(0, 0, 0, 0.2)',
  },
  accent: {
    primary: 'rgb(30, 30, 30)',
    secondary: 'rgb(60, 60, 60)',
    tertiary: 'rgb(90, 90, 90)',
  }
};

const SENTIMENT_COLORS = {
    positive: 'rgba(50, 50, 50, 0.9)',
    neutral: 'rgba(120, 120, 120, 0.7)',
    negative: 'rgba(180, 180, 180, 0.7)'
};

// Modern ChartCard component
const ChartCard = ({ title, description, children, className = "" }) => (
  <Card className={cn("border-2 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] transition-all duration-300", className)}>
    <CardHeader className="pb-4 border-b-2 border-black">
      <div>
        <CardTitle className="text-xl font-bold tracking-tight">{title}</CardTitle>
        {description && <CardDescription className="text-gray-600 mt-1">{description}</CardDescription>}
      </div>
    </CardHeader>
    <CardContent className="pt-6">
      <div className="h-72 relative">
        {children}
      </div>
    </CardContent>
  </Card>
);

// Skeleton component
const SkeletonDashboard = () => (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
    {Array.from({ length: 4 }).map((_, i) => (
      <Card key={i} className="animate-pulse border-2 border-black">
        <CardContent className="p-6">
          <div className="h-5 bg-gray-200 rounded w-1/2 mb-2"></div>
          <div className="h-8 bg-gray-300 rounded w-3/4"></div>
        </CardContent>
      </Card>
    ))}
    <Card className="lg:col-span-4 animate-pulse border-2 border-black">
      <CardHeader>
        <div className="h-5 bg-gray-200 rounded w-1/3"></div>
      </CardHeader>
      <CardContent>
        <div className="h-72 bg-gray-100 rounded-md"></div>
      </CardContent>
    </Card>
  </div>
);

export default function DashboardPage() {
  const [platform, setPlatform] = useState('reddit');
  const [date, setDate] = useState({
    from: addDays(new Date(), -365),
    to: new Date(),
  });
  const [rawData, setRawData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [serverOnline, setServerOnline] = useState(false);

  useEffect(() => {
    const checkServerStatus = async () => {
      try {
        const response = await fetch(`${DASHBOARD_API}/health`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });
        if (response.ok) {
          setServerOnline(true);
          console.log('Server is online');
        } else {
          setServerOnline(false);
          setError('Backend server is not responding');
        }
      } catch (err) {
        console.warn('Server health check failed:', err.message);
        setServerOnline(false);
        setError(`Cannot connect to backend server at ${DASHBOARD_API}`);
      }
    };
    
    checkServerStatus();
  }, []);

  useEffect(() => {
    const fetchDashboardData = async () => {
      if (!date?.from || !date?.to) return;
      
      if (!serverOnline) {
        setError('Backend server is not running. Please start the FastAPI server.');
        setLoading(false);
        return;
      }
      
      setLoading(true);
      setError(null);

      try {
        console.log(`Fetching from: ${DASHBOARD_API}/dashboard`);
        
        const response = await fetch(`${DASHBOARD_API}/dashboard`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            platform: platform,
            date_range: {
              start_date: format(date.from, 'yyyy-MM-dd'),
              end_date: format(date.to, 'yyyy-MM-dd'),
            },
          }),
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
        const data = await response.json();
        console.log('Dashboard data received:', data);
        setRawData(data);
      } catch (err) {
        console.error('Error fetching dashboard:', err);
        setError(`Failed to load data: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };

    if (serverOnline) {
      fetchDashboardData();
    }
  }, [date, platform, serverOnline]);

  const stats = useMemo(() => {
    if (!rawData) return null;

    return {
      totalPosts: rawData.total_posts || 0,
      totalEngagement: rawData.total_engagement || 0,
      totalKeywords: rawData.trending_keywords_count || 0,
      totalUsers: rawData.influential_users_count || 0
    };
  }, [rawData]);

  const platformDoughnutChart = useMemo(() => {
    if (!rawData || !rawData.platform_distribution) return null;
    
    const labels = rawData.platform_distribution.map(p => p.platform);
    const data = rawData.platform_distribution.map(p => p.percentage);
    
    const colors = labels.map((_, idx) => `rgba(${30 + idx * 40}, ${30 + idx * 40}, ${30 + idx * 40}, 0.8)`);
    
    return {
      labels,
      datasets: [{ 
        data, 
        backgroundColor: colors,
        borderColor: '#000',
        borderWidth: 3, 
        cutout: '65%',
        hoverOffset: 8
      }]
    };
  }, [rawData]);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { 
        position: 'bottom', 
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
        displayColors: true,
        callbacks: {
          label: function(context) {
            const label = context.label || '';
            const value = context.parsed;
            return `${label}: ${value.toFixed(1)}%`;
          }
        }
      }
    }
  };

  const formatDateRange = () => {
    if (!date?.from) return "Select date range";
    if (!date?.to) return format(date.from, "MMM dd, yyyy");
    return `${format(date.from, "MMM dd, yyyy")} - ${format(date.to, "MMM dd, yyyy")}`;
  };

  return (
    <div className="space-y-8 p-6 bg-white min-h-screen">
      <header className="border-b-4 border-black pb-6">
        <h1 className="text-4xl font-black tracking-tighter mb-2">Analytics Dashboard</h1>
        <p className="text-gray-600 text-lg">Reddit content analytics and engagement metrics</p>
      </header>

      <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
        <div className="flex items-center gap-3">
          <Popover>
            <PopoverTrigger asChild>
              <Button 
                id="date" 
                variant="outline" 
                className={cn(
                  "w-full sm:w-[300px] justify-start text-left font-semibold border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] transition-all", 
                  !date && "text-gray-400"
                )}
              >
                <CalendarIcon className="mr-2 h-4 w-4" />
                {formatDateRange()}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0 border-2 border-black" align="start">
              <Calendar 
                initialFocus 
                mode="range" 
                defaultMonth={date?.from} 
                selected={date} 
                onSelect={setDate} 
                numberOfMonths={2}
              />
            </PopoverContent>
          </Popover>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-sm font-bold border-2 border-black px-4 py-2 bg-white">
            <Rss className="mr-2 h-4 w-4" />
            Reddit Analytics
          </Badge>
        </div>
      </div>

      {loading ? (
        <SkeletonDashboard />
      ) : error ? (
        <Card className="p-8 border-2 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
          <div className="text-center">
            <AlertCircle className="h-16 w-16 mx-auto mb-4 text-black" />
            <h3 className="text-2xl font-bold mb-3">Connection Error</h3>
            <p className="text-gray-600 mb-6">{error}</p>
            <div className="space-y-3 bg-gray-50 p-6 rounded-lg border-2 border-black">
              <p className="text-sm font-semibold">Troubleshooting Steps:</p>
              <ul className="text-sm text-gray-700 text-left list-disc pl-6 space-y-1">
                <li>Ensure FastAPI server is running on port 8000</li>
                <li>Run: <code className="bg-white px-2 py-1 rounded border border-black font-mono text-xs">python -m app.dashboard_fastapi</code></li>
                <li>Check: <a href="http://localhost:8000/health" target="_blank" className="text-black underline font-semibold hover:no-underline">http://localhost:8000/health</a></li>
              </ul>
            </div>
            <Button 
              variant="outline" 
              onClick={() => window.location.reload()} 
              className="mt-6 border-2 border-black font-bold shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] transition-all"
            >
              Retry Connection
            </Button>
          </div>
        </Card>
      ) : rawData && stats ? (
        <>
          <StatsCardGrid data={stats} />
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <PostsHistogramChart data={rawData} className="lg:col-span-2" />
            
            <ChartCard title="Platform Distribution" description="Reddit content breakdown">
              {platformDoughnutChart ? (
                <Doughnut 
                  options={chartOptions} 
                  data={platformDoughnutChart} 
                />
              ) : (
                <div className="h-full flex items-center justify-center text-gray-400">
                  No platform data available
                </div>
              )}
            </ChartCard>
          </div>

          <EngagementTrendsChart data={rawData} />

          <SentimentTrendsChart data={rawData} />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <PostsList data={rawData} />
            <UsersList data={rawData} />
            <KeywordsList data={rawData} />
          </div>
        </>
      ) : (
        <Card className="p-12 border-2 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
          <div className="text-center text-gray-600">
            <Eye className="h-16 w-16 mx-auto mb-4 text-black" />
            <h3 className="text-2xl font-bold mb-2 text-black">No Data Available</h3>
            <p className="text-lg">Adjust your date range to view analytics data</p>
            {!serverOnline && (
              <p className="text-sm text-black font-semibold mt-4 bg-gray-100 inline-block px-4 py-2 rounded border-2 border-black">
                ⚠️ Backend server appears to be offline
              </p>
            )}
          </div>
        </Card>
      )}
    </div>
  );
}