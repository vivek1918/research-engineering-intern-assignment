'use client';

import React, { useState, useRef, useEffect } from 'react';
import { SendHorizonal } from 'lucide-react'; // Assuming you have lucide-react installed

export function ChatUI() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm your SocialPulse assistant. Ask me a question about your data to generate a report.",
      sender: 'assistant',
    },
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const chatContainerRef = useRef(null);

  // Effect to scroll to the bottom of the chat on new messages
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const handleSend = () => {
    if (input.trim() !== '') {
      // Add user message
      setMessages([...messages, { id: Date.now(), text: input, sender: 'user' }]);
      setInput('');
      setIsTyping(true);

      // Simulate assistant response
      // In a real app, you would make a fetch request to your backend here
      setTimeout(() => {
        setIsTyping(false);
        setMessages((prevMessages) => [
          ...prevMessages,
          { id: Date.now() + 1, text: 'Thank you for your query. I am analyzing the data now.', sender: 'assistant' },
        ]);
      }, 2000);
    }
  };
  
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };
  
  // A simple SVG logo that uses theme colors
  const LogoIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 2L2 7V17L12 22L22 17V7L12 2Z" stroke="hsl(var(--primary-foreground))" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M2 7L12 12L22 7" stroke="hsl(var(--primary-foreground))" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M12 12V22" stroke="hsl(var(--primary-foreground))" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );

  return (
    <div className="flex flex-col h-full max-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="flex items-center p-4 border-b border-border sticky top-0 bg-background/95 backdrop-blur-sm z-10">
         <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary text-primary-foreground">
              <LogoIcon />
            </div>
            <div>
              <h1 className="text-lg font-semibold">SocialPulse AI</h1>
              <p className="text-sm text-muted-foreground">Your data intelligence assistant</p>
            </div>
         </div>
      </header>

      {/* Chat Area */}
      <main ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 chat-container">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex items-end gap-2 ${
              message.sender === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            <div
              className={`max-w-xs md:max-w-md lg:max-w-2xl px-4 py-3 rounded-2xl message-bubble ${
                message.sender === 'user'
                  ? 'user-bubble rounded-br-md'
                  : 'assistant-bubble rounded-bl-md'
              }`}
            >
              <p className="text-sm">{message.text}</p>
            </div>
          </div>
        ))}
        {isTyping && (
          <div className="flex justify-start">
             <div className="typing-indicator">
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
            </div>
          </div>
        )}
      </main>

      {/* Input Area */}
      <footer className="p-4 border-t border-border bg-background sticky bottom-0">
        <div className="relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask for a detailed analysis..."
            className="w-full bg-card border border-border rounded-full py-3 pl-5 pr-14 focus:outline-none focus:ring-2 focus:ring-ring focus:border-ring transition-shadow duration-200 chat-input"
          />
          <button
            onClick={handleSend}
            aria-label="Send message"
            className="absolute right-2 top-1/2 -translate-y-1/2 p-2.5 rounded-full bg-primary text-primary-foreground hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ring"
          >
            <SendHorizonal className="w-5 h-5" />
          </button>
        </div>
      </footer>
    </div>
  );
}
