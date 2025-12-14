// src/app/chatbot/Visualization.jsx

"use client";

import { Bar, Line, Pie } from 'react-chartjs-2';

import {

Chart as ChartJS, CategoryScale, LinearScale, BarElement, LineElement,

PointElement, ArcElement, Title, Tooltip, Legend

} from 'chart.js';


// Register all the necessary components for Chart.js

ChartJS.register(

CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement, Title, Tooltip, Legend

);


const ChartComponent = ({ chartData }) => {

const options = {

responsive: true,

maintainAspectRatio: false,

plugins: {

legend: { position: 'top' },

title: { display: true, text: chartData.title, font: { size: 16, weight: 'bold' }, color: '#1F2937' },

},

scales: {

x: { title: { display: !!chartData.xAxisLabel, text: chartData.xAxisLabel } },

y: { title: { display: !!chartData.yAxisLabel, text: chartData.yAxisLabel } }

}

};


const data = {

labels: chartData.labels,

datasets: [{

label: chartData.yAxisLabel || 'Value',

data: chartData.data,

backgroundColor: chartData.colors || [

'rgba(59, 130, 246, 0.7)', 'rgba(239, 68, 68, 0.7)',

'rgba(245, 158, 11, 0.7)', 'rgba(16, 185, 129, 0.7)',

'rgba(139, 92, 246, 0.7)', 'rgba(236, 72, 153, 0.7)',

],

borderColor: 'rgba(255, 255, 255, 0.5)',

borderWidth: 1,

}],

};


const getChart = () => {

switch(chartData.type) {

case 'line': return <Line options={options} data={data} />;

case 'pie': return <Pie options={options} data={data} />;

case 'bar':

default:

return <Bar options={options} data={data} />;

}

}


return (

<div className="bg-white p-4 rounded-xl shadow-lg border border-gray-200">

<p className="text-sm text-gray-600 mb-4 font-medium">{chartData.description}</p>

<div className="relative h-80">

{getChart()}

</div>

<p className="text-xs text-gray-500 mt-4 italic">**Insight:** {chartData.insights}</p>

</div>

);

};


export default ChartComponent; 