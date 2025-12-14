'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

// Sample data - this will be replaced with data from your backend
const data = [
  { name: 'Jan', tweets: 4000, mentions: 2400 },
  { name: 'Feb', tweets: 3000, mentions: 1398 },
  { name: 'Mar', tweets: 2000, mentions: 9800 },
  { name: 'Apr', tweets: 2780, mentions: 3908 },
  { name: 'May', tweets: 1890, mentions: 4800 },
  { name: 'Jun', tweets: 2390, mentions: 3800 },
  { name: 'Jul', tweets: 3490, mentions: 4300 },
];

export function MyLineChart() {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey="tweets" stroke="#8884d8" />
        <Line type="monotone" dataKey="mentions" stroke="#82ca9d" />
      </LineChart>
    </ResponsiveContainer>
  );
}
