"use client";
import React, { useEffect, useState, useRef } from 'react';
import { cn } from "@/lib/utils";
import { Search } from 'lucide-react';
const API_ENDPOINT = process.env.NEXT_PUBLIC_FASTAPI || "http://127.0.0.1:8000";

// --- Configuration options for the Vis.js network graph ---
const options = {
    layout: {
        hierarchical: false,
    },
    nodes: {
        borderWidth: 2,
        shape: 'dot',
        size: 20,
        font: {
            color: '#1E293B', // slate-800
            size: 14,
            face: 'Inter, sans-serif',
            strokeWidth: 0,
        },
        scaling: {
            min: 20,
            max: 80,
            label: { enabled: true, min: 16, max: 32 }
        },
        shapeProperties: {
            interpolation: false // for diamond shape
        }
    },
    edges: {
        color: {
            color: '#94A3B8', // slate-400
            highlight: '#64748B', // slate-500
            hover: '#475569', // slate-600
            inherit: 'from',
            opacity: 0.8,
        },
        width: 1.5,
        smooth: {
            type: 'continuous'
        }
    },
    physics: {
        forceAtlas2Based: {
            gravitationalConstant: -80,
            centralGravity: 0.01,
            springLength: 230,
            springConstant: 0.08,
            avoidOverlap: 1 
        },
        minVelocity: 0.75,
        solver: 'forceAtlas2Based',
        stabilization: { iterations: 200 }
    },
    interaction: {
        hover: true,
        tooltipDelay: 200,
        navigationButtons: true,
        keyboard: true,
        dragNodes: true,
    },
    groups: {
        user: {
            color: { background: '#E2E8F0', border: '#94A3B8' }, // slate-200 bg, slate-400 border
            font: { size: 14, color: '#1E293B' } // Dark font for light nodes
        },
        cluster: {
            shape: 'diamond',
            color: { background: '#FDE68A', border: '#FBBF24' }, // amber-200 bg, amber-400 border
            font: { size: 18, face: 'Inter, sans-serif', weight: 'bold', color: '#451A03' } // Dark amber font
        },
    },
};

// --- Details Panel Component ---
const DetailsPanel = ({ nodeData, onClose }) => {
    if (!nodeData) return null;

    const isCluster = nodeData.cluster_id && !nodeData.username;

    if (isCluster) {
        return (
            <div className="absolute top-4 right-4 z-20 w-full max-w-md bg-white/80 backdrop-blur-md text-slate-800 border border-slate-200 rounded-lg shadow-2xl font-sans animate-fade-in-right">
                <div className="p-4 flex justify-between items-center border-b border-slate-200">
                    <div className="flex items-center">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-amber-500 mr-3"><path d="m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L12 8.88l-6.64-6.64a1.21 1.21 0 0 0-1.72 0L2.36 3.64a1.21 1.21 0 0 0 0 1.72L8.88 12l-6.52 6.52a1.21 1.21 0 0 0 0 1.72l1.28 1.28a1.21 1.21 0 0 0 1.72 0L12 15.12l6.64 6.64a1.21 1.21 0 0 0 1.72 0l1.28-1.28a1.21 1.21 0 0 0 0-1.72L15.12 12l6.52-6.52a1.21 1.21 0 0 0 0-1.72z"></path></svg>
                        <h3 className="font-bold text-lg text-amber-600">Cluster Details</h3>
                    </div>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-900">&times;</button>
                </div>
                <div className="p-4 text-slate-700 space-y-2">
                    <p><strong>Name:</strong> {nodeData.cluster_name}</p>
                    <p><strong>ID:</strong> <span className="font-mono bg-slate-200 text-slate-800 px-1.5 py-0.5 rounded">{nodeData.cluster_id}</span></p>
                    <p><strong>Items in Cluster:</strong> {nodeData.item_count}</p>
                </div>
                 <div className="p-4 border-t border-slate-200">
                    <h4 className="font-bold text-slate-800 mb-2">Raw Data</h4>
                    <pre className="text-xs bg-slate-100 text-slate-800 rounded p-2 overflow-auto max-h-60">
                        {JSON.stringify(nodeData, null, 2)}
                    </pre>
                 </div>
            </div>
        );
    }

    let title = "No Title";
    if (nodeData.text && typeof nodeData.text === 'string') {
        try {
            const parsedText = JSON.parse(nodeData.text);
            title = parsedText.title || "No Title";
        } catch (e) {
            title = "Title not available";
        }
    }

    return (
        <div className="absolute top-4 right-4 z-20 w-full max-w-md bg-white/80 backdrop-blur-md text-slate-800 border border-slate-200 rounded-lg shadow-2xl font-sans animate-fade-in-right">
             <div className="p-4 flex justify-between items-center border-b border-slate-200">
                 <div className="flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-500 mr-3"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
                    <h3 className="font-bold text-lg text-blue-600">Post Details</h3>
                </div>
                <button onClick={onClose} className="text-gray-500 hover:text-gray-900 text-2xl font-bold">&times;</button>
            </div>
            <div className="p-4 text-slate-700 space-y-3">
                <p className="mb-2 text-slate-800 italic">"{title}"</p>
                <p><strong>User:</strong> <span className="text-blue-600">{nodeData.username || 'N/A'}</span></p>
                <p><strong>Platform:</strong> {nodeData.platform || 'N/A'}</p>
                <p><strong>Belongs to Cluster:</strong> <span className="text-amber-600">{nodeData.cluster_name || 'N/A'}</span></p>
                <hr className="border-slate-200 my-2" />
                <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                    <p><strong>Engagement:</strong> {nodeData.engagement ?? '0'}</p>
                    <p><strong>Views:</strong> {nodeData.views ?? '0'}</p>
                    <p><strong>Comments:</strong> {nodeData.comments ?? '0'}</p>
                    <p><strong>Reposts:</strong> {nodeData.reposts ?? '0'}</p>
                </div>
                {nodeData.link && <a href={nodeData.link} target="_blank" rel="noopener noreferrer" className="inline-block pt-2 text-blue-600 hover:text-blue-800 hover:underline">View Source Post &rarr;</a>}
            </div>
             <div className="p-4 border-t border-slate-200">
                <h4 className="font-bold text-slate-800 mb-2">Raw Data</h4>
                <pre className="text-xs bg-slate-100 text-slate-800 rounded p-2 overflow-auto max-h-48">
                    {JSON.stringify(nodeData, null, 2)}
                </pre>
            </div>
        </div>
    );
};

// --- Chat Response Modal Component ---
const ChatResponseModal = ({ response, isLoading, error, onClose, query }) => (
    <div className="fixed inset-0 bg-black/30 z-50 flex items-center justify-center p-4 animate-fade-in">
        <div className="bg-white/80 backdrop-blur-md rounded-lg shadow-2xl w-full max-w-2xl max-h-[80vh] flex flex-col border border-slate-200">
            <div className="p-4 flex justify-between items-center border-b border-slate-200">
                <h3 className="font-bold text-lg text-sky-600">Query Response</h3>
                <button onClick={onClose} className="text-gray-500 hover:text-gray-900 text-2xl font-bold">&times;</button>
            </div>
            <div className="p-6 overflow-y-auto">
                <div className="mb-4 p-3 bg-slate-100 rounded-md">
                    <p className="text-sm text-slate-500 mb-1">Your query:</p>
                    <p className="font-mono text-slate-800">{query}</p>
                </div>
                {isLoading && (
                    <div className="flex items-center justify-center gap-3 text-slate-600">
                       <svg className="animate-spin h-6 w-6 text-sky-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                        <span>Thinking...</span>
                    </div>
                )}
                {error && <p className="text-red-700 bg-red-100 p-3 rounded-md">{error}</p>}
                {response && !isLoading && <div className="text-slate-800 whitespace-pre-wrap leading-relaxed">{response}</div>}
            </div>
        </div>
    </div>
);

const SkeletonLoader = () => (
    <div className="space-y-6 h-full flex flex-col">
        <header>
            <div className="h-8 bg-slate-200 rounded w-1/3 mb-2 animate-pulse"></div>
            <div className="h-4 bg-slate-200 rounded w-1/2 animate-pulse"></div>
        </header>
        <div className="flex-1 bg-slate-100 rounded-lg animate-pulse"></div>
        <footer className="w-full max-w-4xl mx-auto pb-2 flex-shrink-0">
            <div className="h-12 bg-slate-200 rounded-xl animate-pulse"></div>
        </footer>
    </div>
);


export default function VisualizePage() {
    const [graphData, setGraphData] = useState(null);
    const [clusterNames, setClusterNames] = useState([]);
    const [network, setNetwork] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [visJsLoaded, setVisJsLoaded] = useState(false);
    const [selectedNodeData, setSelectedNodeData] = useState(null);
    const [inputValue, setInputValue] = useState('');
    const [searchError, setSearchError] = useState('');
    
    // Autocomplete state
    const [suggestions, setSuggestions] = useState([]);
    const [showSuggestions, setShowSuggestions] = useState(false);
    
    const [showChatModal, setShowChatModal] = useState(false);
    const [isChatLoading, setIsChatLoading] = useState(false);
    const [chatResponse, setChatResponse] = useState('');
    const [chatError, setChatError] = useState(null);
    const [submittedQuery, setSubmittedQuery] = useState('');

    const containerRef = useRef(null);
    const inputRef = useRef(null); // Ref for the input element

    useEffect(() => {
        if (window.vis) { setVisJsLoaded(true); return; }
        const cssLink = document.createElement('link');
        cssLink.href = 'https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css';
        cssLink.rel = 'stylesheet'; cssLink.type = 'text/css';
        document.head.appendChild(cssLink);
        
        const script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js';
        script.async = true;
        script.onload = () => setVisJsLoaded(true);
        script.onerror = () => setError("Visualization library (vis.js) failed to load.");
        document.body.appendChild(script);

        return () => {
            if (document.head.contains(cssLink)) document.head.removeChild(cssLink);
            if (document.body.contains(script)) document.body.removeChild(script);
        };
    }, []);

    useEffect(() => {
        if (!visJsLoaded) return;
        const fetchDataAndBuildGraph = async () => {
            setLoading(true); setError(null);
            try {
                await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate loading
                const response = await fetch(`${API_ENDPOINT}/cluster`, { method: 'POST' });
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                const result = await response.json();
                
                setClusterNames(result.clusters.map(c => c.cluster_name));
                
                const allClusterNodes = result.clusters.map(c => ({ id: `cluster_${c.cluster_id}`, label: `${c.cluster_name}\n(${c.item_count})`, group: 'cluster', value: c.item_count, fullData: c }));
                const allUserNodes = [];
                const allEdges = [];
                const userNodeIds = new Set();
                const clusterNodeIds = new Set(result.clusters.map(c => `cluster_${c.cluster_id}`));

                result.data.forEach((item, index) => {
                    const username = item.username || item.author || 'Unknown User';
                    const uniqueId = item.data?.id || item.id || `item_${index}`;
                    const postNodeId = `user_${username}_${uniqueId}`;
                    
                    if (username && !userNodeIds.has(postNodeId)) {
                        allUserNodes.push({ id: postNodeId, label: username, group: 'user', value: 1, fullData: item });
                        userNodeIds.add(postNodeId);
                    }
                    
                    const clusterNodeId = `cluster_${item.cluster_id}`;
                    if (clusterNodeIds.has(clusterNodeId)) {
                        const edgeId = `edge_${postNodeId}_${clusterNodeId}`;
                        allEdges.push({ id: edgeId, from: postNodeId, to: clusterNodeId });
                    }
                });
                setGraphData({ nodes: new window.vis.DataSet([...allClusterNodes, ...allUserNodes]), edges: new window.vis.DataSet(allEdges) });
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };
        fetchDataAndBuildGraph();
    }, [visJsLoaded]);

    useEffect(() => {
        if (containerRef.current && graphData && !network) {
            const visNetwork = new window.vis.Network(containerRef.current, graphData, options);
            visNetwork.on('click', (params) => {
                if (params.nodes.length > 0) {
                    const nodeId = params.nodes[0];
                    const clickedNode = graphData.nodes.get(nodeId);
                    setSelectedNodeData(clickedNode.fullData);
                } else {
                    setSelectedNodeData(null);
                    visNetwork.unselectAll();
                }
            });
            setNetwork(visNetwork);
        }
    }, [graphData, network]);

    const handleInputChange = (e) => {
        const value = e.target.value;
        setInputValue(value);
        if (value.startsWith('@') && !value.includes(' ')) {
            const searchTerm = value.substring(1).toLowerCase();
            const filtered = clusterNames.filter(name => name.toLowerCase().includes(searchTerm));
            setSuggestions(filtered);
            setShowSuggestions(true);
        } else {
            setShowSuggestions(false);
        }
    };
    
    const handleSuggestionClick = (name) => {
        setInputValue(`@${name} `);
        setShowSuggestions(false);
        setTimeout(() => inputRef.current?.focus(), 0);
    };

    const handleQuerySubmit = async () => {
        setSearchError('');
        const text = inputValue.trim();
        if (!text) return;
        setShowSuggestions(false);
    
        if (text.startsWith('@')) {
            const queryWithoutAt = text.substring(1);
            let identifiedCluster = null;
    
            // Sort cluster names by length descending to find the longest match first
            const sortedClusterNames = [...clusterNames].sort((a, b) => b.length - a.length);
    
            for (const name of sortedClusterNames) {
                if (queryWithoutAt.toLowerCase().startsWith(name.toLowerCase())) {
                    identifiedCluster = name;
                    break; // Found the longest possible match
                }
            }
    
            if (!identifiedCluster) {
                setSearchError("Could not identify a valid cluster. Please select from suggestions.");
                return;
            }
    
            const questionText = queryWithoutAt.substring(identifiedCluster.length).trim();
    
            if (!questionText) {
                setSearchError(`Please provide a question after '@${identifiedCluster}'.`);
                return;
            }
    
            const clusterName = identifiedCluster;
    
            if (graphData && network) {
                const matchedNodes = graphData.nodes.get({
                    filter: (item) => item.group === 'cluster' && item.fullData?.cluster_name === clusterName
                });
    
                if (matchedNodes.length > 0) {
                    const nodeId = matchedNodes[0].id;
                    network.focus(nodeId, { scale: 1.5, animation: true });
                    network.selectNodes([nodeId]);
                    setSelectedNodeData(matchedNodes[0].fullData);
                }
            }
    
            setSubmittedQuery(text);
            setShowChatModal(true);
            setIsChatLoading(true);
            setChatError(null);
            setChatResponse('');
            try {
                const res = await fetch(`${API_ENDPOINT}/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cluster_name: clusterName, question: questionText }) });
                if (!res.ok) { const err = await res.json(); throw new Error(err.detail || `Server error: ${res.status}`); }
                const result = await res.json();
                setChatResponse(result.response);
            } catch (err) {
                setChatError(err.message);
            } finally {
                setIsChatLoading(false);
            }
        } else {
            if (!graphData || !network) return;
            const normalizedText = text.replace(/\s+/g, '').toLowerCase();
            const matchedNodes = graphData.nodes.get({
                filter: (item) => item.group === 'cluster' && (item.fullData?.cluster_name.replace(/\s+/g, '').toLowerCase().includes(normalizedText))
            });
            if (matchedNodes.length > 0) {
                const nodeId = matchedNodes[0].id;
                network.focus(nodeId, { scale: 1.5, animation: true });
                network.selectNodes([nodeId]);
                setSelectedNodeData(matchedNodes[0].fullData);
            } else {
                setSearchError(`Cluster "${text}" not found.`);
            }
        }
    };
    
    if (loading) {
        return <SkeletonLoader />;
    }

    return (
        <div className="space-y-6 h-full flex flex-col font-sans">
            {showChatModal && <ChatResponseModal response={chatResponse} isLoading={isChatLoading} error={chatError} onClose={() => setShowChatModal(false)} query={submittedQuery} />}
            <header>
                <h1 className="text-2xl font-bold tracking-tight text-foreground">User & Topic Cluster Analysis</h1>
                <p className="text-muted-foreground">Click nodes for details or use '@' to query a cluster.</p>
            </header>

            <div className="flex-1 relative bg-card border rounded-lg shadow-sm p-4">
                {error ? (
                    <div className="w-full h-full flex items-center justify-center p-6">
                        <div className="bg-red-100 text-red-800 px-6 py-4 rounded-lg shadow-lg flex items-center gap-4 border border-red-200">
                            <div><h2 className="text-lg font-bold">An Error Occurred</h2><p className="font-mono text-sm">{error}</p></div>
                        </div>
                    </div>
                ) : (
                    <>
                        <div ref={containerRef} className="w-full h-full" />
                        <DetailsPanel nodeData={selectedNodeData} onClose={() => setSelectedNodeData(null)} />
                    </>
                )}
            </div>
            
            <footer className="w-full max-w-4xl mx-auto pb-2 flex-shrink-0">
                <div className="relative">
                    {showSuggestions && suggestions.length > 0 && (
                        <div className="absolute bottom-full mb-2 w-full bg-white border rounded-lg shadow-lg z-30 max-h-48 overflow-y-auto">
                            {suggestions.map(name => (
                                <button key={name} onClick={() => handleSuggestionClick(name)} className="w-full text-left px-4 py-2 hover:bg-slate-100">
                                    @{name}
                                </button>
                            ))}
                        </div>
                    )}
                    {searchError && <p className="text-center text-red-700 text-sm mb-2">{searchError}</p>}
                    <div className="flex gap-2">
                        <div className="relative w-full">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                            <input
                              ref={inputRef}
                              type="text"
                              value={inputValue}
                              onChange={handleInputChange}
                              onKeyUp={(e) => e.key === 'Enter' && handleQuerySubmit()}
                              placeholder="Search cluster or type '@cluster-name query...'"
                              className="w-full bg-background border shadow-inner rounded-xl py-3 pl-10 pr-4 focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all text-base"
                            />
                        </div>
                        <button
                          onClick={handleQuerySubmit}
                          className="bg-primary text-primary-foreground hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary transition-all shadow-sm rounded-xl px-6 py-3 font-semibold text-base"
                        >
                            Submit
                        </button>
                    </div>
                </div>
            </footer>
             <style jsx global>{`
                .vis-network, .vis-network canvas { border: none !important; outline: none !important; }
                @keyframes fade-in-right { from { opacity: 0; transform: translateX(20px); } to { opacity: 1; transform: translateX(0); } }
                .animate-fade-in-right { animation: fade-in-right 0.3s ease-out forwards; }
                @keyframes fade-in { from { opacity: 0; } to { opacity: 1; } }
                .animate-fade-in { animation: fade-in 0.2s ease-out forwards; }
             `}</style>
        </div>
    );
}

