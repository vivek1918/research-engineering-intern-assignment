
"use client";

import { useState, useRef } from 'react';

import {

BarChart3,

FileText,

Lightbulb,

CheckCircle,

Info,

BrainCircuit,

Activity,

ChevronDown,

ChevronRight,

Eye,

EyeOff,

TrendingUp,

Target,

Zap,

Download,

FileDown,

AlertCircle

} from 'lucide-react';

import ChartComponent from './Visualization';


// Utility function to return text as is (number highlighting removed)

const highlightNumbers = (text) => {

return text;

};


// Function to convert text to bullet points

const createBulletPoints = (text) => {

if (!text) return [];

// Split by sentences and create meaningful bullet points

const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 10);

return sentences.map(s => s.trim()).filter(Boolean);

};


// Enhanced PDF Export functionality

const exportToPDF = async (data, setProgress) => {

try {

setProgress(10);

// Dynamically import the libraries to reduce bundle size

const html2canvas = (await import('html2canvas')).default;

const { jsPDF } = await import('jspdf');

setProgress(20);


// Force all collapsible sections to be open for the PDF

const collapsibleSections = document.querySelectorAll('[data-collapsible="true"]');

const originalStates = [];

// Store original states and expand all sections

collapsibleSections.forEach((section, index) => {

const button = section.querySelector('button');

const content = section.querySelector('[data-collapsible-content="true"]');

if (button && content) {

originalStates[index] = !content.classList.contains('hidden') && content.style.display !== 'none';

// Expand the section if it's collapsed

if (content.style.display === 'none' || content.classList.contains('hidden')) {

button.click();

}

}

});


setProgress(30);


// Wait for any animations, chart renders, or state changes to complete

await new Promise(resolve => setTimeout(resolve, 1000));


setProgress(40);


// Get the report element

const reportElement = document.getElementById('pdf-export-content');

if (!reportElement) {

throw new Error('Report content not found');

}


// Ensure all images and charts are loaded

const images = reportElement.querySelectorAll('img, canvas, svg');

await Promise.all(Array.from(images).map(img => {

if (img.complete !== undefined) {

return img.complete ? Promise.resolve() : new Promise(resolve => {

img.onload = resolve;

img.onerror = resolve; // Continue even if image fails to load

});

}

return Promise.resolve();

}));


setProgress(50);


// Configure html2canvas options for high quality

const canvas = await html2canvas(reportElement, {

allowTaint: true,

useCORS: true,

backgroundColor: '#f9fafb', // Match your background color

scale: 2, // Higher resolution for better quality

logging: false,

width: reportElement.scrollWidth,

height: reportElement.scrollHeight,

scrollX: 0,

scrollY: 0,

windowWidth: 1400, // Fixed width for consistent rendering

windowHeight: reportElement.scrollHeight,

onclone: (clonedDoc) => {

// Ensure all styles are properly applied in the cloned document

const clonedElement = clonedDoc.getElementById('pdf-export-content');

if (clonedElement) {

clonedElement.style.width = '1400px';

clonedElement.style.maxWidth = 'none';

// Ensure charts are visible in clone

const charts = clonedElement.querySelectorAll('canvas, svg');

charts.forEach(chart => {

chart.style.maxWidth = '100%';

chart.style.height = 'auto';

});

}

}

});


setProgress(70);


// Calculate PDF dimensions

const imgData = canvas.toDataURL('image/jpeg', 0.95);

const pdf = new jsPDF('p', 'mm', 'a4');

const pdfWidth = pdf.internal.pageSize.getWidth();

const pdfHeight = pdf.internal.pageSize.getHeight();

const imgWidth = canvas.width;

const imgHeight = canvas.height;

// Calculate scaling to fit width with margins

const margin = 10; // 10mm margin

const availableWidth = pdfWidth - (margin * 2);

const availableHeight = pdfHeight - (margin * 2);

const ratio = Math.min(availableWidth / imgWidth, availableHeight / imgHeight);

const scaledWidth = imgWidth * ratio;

const scaledHeight = imgHeight * ratio;


setProgress(80);

// Handle multi-page PDF if content is too long

let yPosition = 0;

let pageCount = 1;

while (yPosition < scaledHeight) {

const remainingHeight = scaledHeight - yPosition;

const pageHeight = Math.min(availableHeight, remainingHeight);

// Add new page if not the first page

if (yPosition > 0) {

pdf.addPage();

pageCount++;

}

// Calculate the source rectangle for this page

const srcY = yPosition / ratio;

const srcHeight = pageHeight / ratio;

// Create a temporary canvas for this page

const pageCanvas = document.createElement('canvas');

pageCanvas.width = imgWidth;

pageCanvas.height = srcHeight;

const pageCtx = pageCanvas.getContext('2d');

pageCtx.drawImage(

canvas,

0, srcY, imgWidth, srcHeight, // source rectangle

0, 0, imgWidth, srcHeight // destination rectangle

);

const pageImgData = pageCanvas.toDataURL('image/jpeg', 0.95);

// Add image to PDF

pdf.addImage(

pageImgData,

'JPEG',

margin,

margin,

scaledWidth,

pageHeight

);

yPosition += pageHeight;

}


setProgress(90);


// Add metadata

const timestamp = new Date().toISOString().split('T')[0];

const timeString = new Date().toLocaleTimeString('en-US', {

hour12: false,

hour: '2-digit',

minute: '2-digit'

});

pdf.setProperties({

title: `AI Analysis Report - ${timestamp}`,

subject: 'AI Analysis Report',

author: 'SocialPulse',

creator: 'SocialPulse Assistant',

producer: 'SocialPulse',

keywords: 'AI, Analysis, Report, Data'

});


// Add footer to each page

const totalPages = Math.ceil(scaledHeight / availableHeight);

for (let i = 1; i <= totalPages; i++) {

pdf.setPage(i);

pdf.setFontSize(8);

pdf.setTextColor(128, 128, 128);

pdf.text(

`Generated on ${timestamp} at ${timeString} | Page ${i} of ${totalPages}`,

margin,

pdfHeight - 5

);

}


setProgress(95);


// Save the PDF

pdf.save(`AI_Analysis_Report_${timestamp}_${timeString.replace(':', '-')}.pdf`);

setProgress(100);


// Show success message

setTimeout(() => {

alert('PDF report has been successfully generated and downloaded!');

}, 500);

return true;

} catch (error) {

console.error('Error generating PDF:', error);

// Show detailed error message

let errorMessage = 'An error occurred while generating the PDF. ';

if (error.message.includes('html2canvas')) {

errorMessage += 'There was an issue capturing the report content. Please try again.';

} else if (error.message.includes('jsPDF')) {

errorMessage += 'There was an issue creating the PDF file. Please try again.';

} else {

errorMessage += `Details: ${error.message}`;

}

alert(errorMessage);

return false;

} finally {

setProgress(0);

// Restore original states of collapsible sections

const collapsibleSections = document.querySelectorAll('[data-collapsible="true"]');

const originalStates = JSON.parse(sessionStorage.getItem('originalStates') || '[]');

collapsibleSections.forEach((section, index) => {

const button = section.querySelector('button');

const content = section.querySelector('[data-collapsible-content="true"]');

if (button && content && originalStates[index] === false) {

// Collapse the section if it was originally collapsed

setTimeout(() => button.click(), 100);

}

});

// Clean up session storage

sessionStorage.removeItem('originalStates');

}

};


// Enhanced Download Report Button Component

const DownloadReportButton = ({ data, isGenerating, setIsGenerating, progress, setProgress }) => {

const [showTooltip, setShowTooltip] = useState(false);

const handleDownload = async () => {

setIsGenerating(true);

// Store original states

const collapsibleSections = document.querySelectorAll('[data-collapsible="true"]');

const originalStates = [];

collapsibleSections.forEach((section, index) => {

const content = section.querySelector('[data-collapsible-content="true"]');

originalStates[index] = content && !content.classList.contains('hidden') && content.style.display !== 'none';

});

sessionStorage.setItem('originalStates', JSON.stringify(originalStates));

try {

await exportToPDF(data, setProgress);

} finally {

setIsGenerating(false);

setProgress(0);

}

};


return (

<div className="relative">

<button

onClick={handleDownload}

disabled={isGenerating}

className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white rounded-lg shadow-sm transition-all duration-200 font-medium text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"

aria-label="Export report as PDF"

onMouseEnter={() => setShowTooltip(true)}

onMouseLeave={() => setShowTooltip(false)}

>

{isGenerating ? (

<>

<div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>

<span>Generating... {progress}%</span>

</>

) : (

<>

<FileDown className="w-4 h-4" />

Export PDF

</>

)}

</button>

{/* Progress bar */}

{isGenerating && progress > 0 && (

<div className="absolute top-full left-0 right-0 mt-2 bg-gray-200 rounded-full h-2 overflow-hidden">

<div

className="bg-blue-600 h-full transition-all duration-300 ease-out"

style={{ width: `${progress}%` }}

></div>

</div>

)}

{/* Tooltip */}

{showTooltip && !isGenerating && (

<div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 text-sm text-white bg-gray-800 rounded-lg whitespace-nowrap">

Download complete report with all charts and data

<div className="absolute top-full left-1/2 transform -translate-x-1/2 w-2 h-2 bg-gray-800 rotate-45"></div>

</div>

)}

</div>

);

};


// Enhanced StatCard with bullet points

const StatCard = ({ title, value, insight, icon: Icon, color = "blue" }) => {

const bulletPoints = createBulletPoints(insight);

const colorClasses = {

blue: "bg-blue-50 border-blue-200 text-blue-700",

green: "bg-green-50 border-green-200 text-green-700",

orange: "bg-orange-50 border-orange-200 text-orange-700"

};


return (

<div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm transition-all hover:shadow-md hover:border-blue-300 h-full">

<div className="flex items-center gap-3 mb-4">

{Icon && (

<div className={`w-8 h-8 rounded-lg ${colorClasses[color]} flex items-center justify-center`}>

<Icon className="w-4 h-4" />

</div>

)}

<div>

<h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider">{title}</h3>

<p className="text-xl font-bold text-gray-800">{value}</p>

</div>

</div>

{/* Bullet Points */}

<div className="space-y-3">

{bulletPoints.length > 0 ? (

<ul className="space-y-2">

{bulletPoints.map((point, index) => (

<li key={index} className="flex items-start gap-2 text-sm text-gray-600 leading-relaxed">

<span className="w-1.5 h-1.5 bg-blue-400 rounded-full mt-2 flex-shrink-0"></span>

<span>{highlightNumbers(point)}</span>

</li>

))}

</ul>

) : (

<div className="text-sm text-gray-600 leading-relaxed">

{highlightNumbers(insight)}

</div>

)}

</div>

</div>

);

};


// Collapsible Section Component

const CollapsibleSection = ({

title,

children,

defaultOpen = true,

icon: Icon,

badge,

description

}) => {

const [isOpen, setIsOpen] = useState(defaultOpen);


return (

<section className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden" data-collapsible="true">

<button

onClick={() => setIsOpen(!isOpen)}

className="w-full flex items-center justify-between p-6 text-left hover:bg-gray-50 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-inset"

>

<div className="flex items-center gap-3">

{Icon && <Icon className="w-6 h-6 text-blue-600" />}

<div>

<div className="flex items-center gap-2">

<h2 className="text-2xl font-bold text-gray-800">{title}</h2>

{badge && (

<span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">

{badge}

</span>

)}

</div>

{description && (

<p className="text-sm text-gray-500 mt-1">{description}</p>

)}

</div>

</div>

<div className="flex items-center gap-2 text-gray-400">

<span className="text-sm font-medium">

{isOpen ? 'See less' : 'See more'}

</span>

{isOpen ? (

<ChevronDown className="w-5 h-5" />

) : (

<ChevronRight className="w-5 h-5" />

)}

</div>

</button>

<div

className={`px-6 pb-6 border-t border-gray-100 bg-gray-50/30 ${isOpen ? '' : 'hidden'}`}

data-collapsible-content="true"

>

<div className="pt-6">

{children}

</div>

</div>

</section>

);

};


// Individual Chart Section Component

const ChartSection = ({ chart, index }) => {

const [isOpen, setIsOpen] = useState(false);

const chartRef = useRef(null);


return (

<div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden" data-collapsible="true">

<button

onClick={() => setIsOpen(!isOpen)}

className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-inset"

>

<div className="flex items-center gap-3">

<div className="w-8 h-8 rounded-lg bg-blue-100 flex items-center justify-center">

<BarChart3 className="w-4 h-4 text-blue-600" />

</div>

<div>

<h3 className="font-semibold text-gray-800">{chart.title}</h3>

<p className="text-sm text-gray-500">{chart.type?.toUpperCase()} Chart</p>

</div>

</div>

<div className="flex items-center gap-2 text-gray-400">

{isOpen ? (

<EyeOff className="w-4 h-4" />

) : (

<Eye className="w-4 h-4" />

)}

</div>

</button>

<div

className={`p-4 border-t border-gray-100 bg-gray-50/30 ${isOpen ? '' : 'hidden'}`}

data-collapsible-content="true"

ref={chartRef}

>

<div className="min-h-[400px]"> {/* Ensure minimum height for charts */}

<ChartComponent chartData={chart} />

</div>

{chart.description && (

<p className="text-sm text-gray-600 mt-3 leading-relaxed">

{highlightNumbers(chart.description)}

</p>

)}

{chart.insights && (

<div className="mt-3 p-3 bg-blue-50 rounded-lg border border-blue-200">

<div className="flex items-start gap-2">

<Lightbulb className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />

<p className="text-sm text-gray-700">

<span className="font-medium">Key Insight:</span> {highlightNumbers(chart.insights)}

</p>

</div>

</div>

)}

</div>

</div>

);

};


const ReportDisplay = ({ data }) => {

const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);

const [pdfProgress, setPdfProgress] = useState(0);


if (!data || !data.report) {

return (

<div className="bg-white rounded-lg p-8 text-center">

<AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />

<p className="text-gray-600">No report data available.</p>

</div>

);

}


const { summary, report, visualizations } = data;


return (

<div className="bg-gray-50 rounded-lg p-4 sm:p-6 space-y-6 max-w-full relative">

{/* Export PDF Button - Fixed Position */}

<div className="absolute top-4 right-4 z-10">

<DownloadReportButton

data={data}

isGenerating={isGeneratingPDF}

setIsGenerating={setIsGeneratingPDF}

progress={pdfProgress}

setProgress={setPdfProgress}

/>

</div>


{/* Main Report Content - This will be captured for PDF */}

<div id="pdf-export-content" className="space-y-6 bg-white p-6 rounded-lg">

{/* AI Analysis Summary Section - Always Visible */}

<section>

<div className="flex items-center gap-3 mb-6">

<div className="w-10 h-10 rounded-xl bg-blue-100 flex items-center justify-center">

<BrainCircuit className="w-6 h-6 text-blue-600"/>

</div>

<div>

<h1 className="text-3xl font-bold text-gray-800">AI Analysis Summary</h1>

<p className="text-gray-500 text-sm">Key insights at a glance</p>

</div>

</div>

<div className="grid grid-cols-1 md:grid-cols-3 gap-6">

<StatCard

title="Overview"

value="Key Trends"

insight={summary?.overview || "No overview data available"}

icon={TrendingUp}

color="blue"

/>

<StatCard

title="Key Metrics"

value="Top Figures"

insight={summary?.key_metrics || "No metrics data available"}

icon={Target}

color="green"

/>

<StatCard

title="Primary Insight"

value="Main Takeaway"

insight={summary?.primary_insight || "No insights available"}

icon={Zap}

color="orange"

/>

</div>

</section>


{/* Detailed Report Section - Collapsible */}

<CollapsibleSection

title="Detailed Report"

icon={FileText}

badge={`${(report.key_findings?.length || 0) + (report.actionable_insights?.length || 0)} insights`}

description="Comprehensive analysis with findings and recommendations"

defaultOpen={false}

>

<div className="prose prose-sm sm:prose-base max-w-none text-gray-700 space-y-6">

{/* Analysis Text */}

{report.detailed_analysis && (

<div>

<div className="flex items-center gap-2 mb-3">

<Activity className="w-5 h-5 text-blue-600" />

<h3 className="text-lg font-semibold text-gray-800 m-0">Analysis</h3>

</div>

<div className="bg-gray-50 p-4 rounded-lg border-l-4 border-blue-400">

<p className="leading-relaxed m-0">

{highlightNumbers(report.detailed_analysis)}

</p>

</div>

</div>

)}


{/* Key Findings */}

{report.key_findings && report.key_findings.length > 0 && (

<div>

<div className="flex items-center gap-2 mb-4">

<CheckCircle className="w-5 h-5 text-green-600" />

<h3 className="text-lg font-semibold text-gray-800 m-0">

Key Findings ({report.key_findings.length})

</h3>

</div>

<div className="space-y-3">

{report.key_findings.map((item, index) => (

<div key={index} className="flex gap-3 p-4 bg-green-50 rounded-lg border border-green-200 hover:bg-green-100 transition-colors">

<div className="w-6 h-6 bg-green-100 text-green-700 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0 mt-0.5">

{index + 1}

</div>

<p className="text-gray-700 leading-relaxed m-0">

{highlightNumbers(item)}

</p>

</div>

))}

</div>

</div>

)}


{/* Actionable Insights */}

{report.actionable_insights && report.actionable_insights.length > 0 && (

<div>

<div className="flex items-center gap-2 mb-4">

<Lightbulb className="w-5 h-5 text-orange-600" />

<h3 className="text-lg font-semibold text-gray-800 m-0">

Actionable Insights ({report.actionable_insights.length})

</h3>

</div>

<div className="space-y-3">

{report.actionable_insights.map((item, index) => (

<div key={index} className="flex gap-3 p-4 bg-orange-50 rounded-lg border border-orange-200 hover:bg-orange-100 transition-colors">

<div className="w-6 h-6 bg-orange-100 text-orange-700 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">

<Lightbulb className="w-3 h-3" />

</div>

<p className="text-gray-700 leading-relaxed m-0">

{highlightNumbers(item)}

</p>

</div>

))}

</div>

</div>

)}


{/* Methodology and Data Quality */}

{(report.methodology || report.data_quality_notes) && (

<div className="grid md:grid-cols-2 gap-4 pt-4 border-t border-gray-200">

{report.methodology && (

<div className="p-4 bg-blue-50 rounded-lg border border-blue-200">

<div className="flex items-center gap-2 mb-2">

<Info className="w-4 h-4 text-blue-600" />

<h4 className="font-semibold text-gray-800">Methodology</h4>

</div>

<p className="text-sm text-gray-600 leading-relaxed">

{highlightNumbers(report.methodology)}

</p>

</div>

)}

{report.data_quality_notes && (

<div className="p-4 bg-yellow-50 rounded-lg border border-yellow-200">

<div className="flex items-center gap-2 mb-2">

<CheckCircle className="w-4 h-4 text-yellow-600" />

<h4 className="font-semibold text-gray-800">Data Quality</h4>

</div>

<p className="text-sm text-gray-600 leading-relaxed">

{highlightNumbers(report.data_quality_notes)}

</p>

</div>

)}

</div>

)}

</div>

</CollapsibleSection>


{/* Visualizations Section - Collapsible */}

{visualizations && visualizations.length > 0 && (

<CollapsibleSection

title="Visualizations"

icon={BarChart3}

badge={`${visualizations.length} charts`}

description="Interactive charts and graphs"

defaultOpen={false}

>

<div className="space-y-4">

{visualizations.map((chart, index) => (

<ChartSection key={chart.id || index} chart={chart} index={index} />

))}

</div>

</CollapsibleSection>

)}

</div>


{/* Loading Overlay */}

{isGeneratingPDF && (

<div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">

<div className="bg-white p-6 rounded-lg shadow-lg max-w-sm w-full mx-4">

<div className="flex items-center gap-3 mb-4">

<div className="w-8 h-8 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>

<h3 className="font-semibold text-gray-800">Generating PDF Report</h3>

</div>

<div className="space-y-2">

<div className="flex justify-between text-sm text-gray-600">

<span>Progress</span>

<span>{pdfProgress}%</span>

</div>

<div className="bg-gray-200 rounded-full h-2 overflow-hidden">

<div

className="bg-blue-600 h-full transition-all duration-300 ease-out"

style={{ width: `${pdfProgress}%` }}

></div>

</div>

<p className="text-xs text-gray-500 mt-2">

{pdfProgress < 30 ? 'Preparing content...' :

pdfProgress < 60 ? 'Capturing visuals...' :

pdfProgress < 90 ? 'Generating PDF...' : 'Finalizing...'}

</p>

</div>

</div>

</div>

)}

</div>

);

};


export default ReportDisplay;


