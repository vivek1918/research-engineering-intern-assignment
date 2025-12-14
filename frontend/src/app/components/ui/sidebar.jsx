// src/app/components/ui/sidebar.jsx
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  MessageSquare,
  BarChart3,
  Newspaper,
  ChevronLeft,
  ChevronRight,
  Bot,
  User,
  LineChart,
} from "lucide-react";
import { Button } from "@/app/components/ui/button";
import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

const sidebarItems = [
  {
    name: "Chatbot",
    href: "/chatbot",
    icon: MessageSquare,
  },
  {
    name: "Dashboard",
    href: "/dashboard",
    icon: BarChart3,
  },
  {
    name: "News",
    href: "/news",
    icon: Newspaper,
  },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [isCollapsed, setIsCollapsed] = useState(false);

  const toggleSidebar = () => {
    setIsCollapsed(!isCollapsed);
  };

  return (
    <TooltipProvider delayDuration={0}>
      <div
        className={cn(
          "relative h-screen bg-background text-foreground border-r transition-all duration-300 ease-in-out flex flex-col dark",
          isCollapsed ? "w-20" : "w-64"
        )}
      >
        {/* Toggle Button */}
        <div className="absolute -right-3 top-8 z-50">
          <Button
            variant="outline"
            size="icon"
            className="h-6 w-6 rounded-full bg-background hover:bg-muted"
            onClick={toggleSidebar}
          >
            {isCollapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </Button>
        </div>

        {/* Logo */}
        <div className="flex items-center p-4 border-b h-16">
          <Bot
            className={cn(
              "h-8 w-8 text-primary transition-all",
              isCollapsed ? "mx-auto" : "mr-3"
            )}
          />
          <h1
            className={cn(
              "text-xl font-bold whitespace-nowrap transition-opacity duration-200",
              isCollapsed ? "opacity-0" : "opacity-100"
            )}
          >
            SocialPulse
          </h1>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 flex flex-col gap-1 p-2 mt-4">
          {sidebarItems.map((item) => {
            const isActive = pathname.startsWith(item.href);

            const linkContent = (
              <>
                <item.icon
                  className={cn(
                    "h-5 w-5 flex-shrink-0",
                    isActive && "text-primary-foreground"
                  )}
                />
                <span
                  className={cn(
                    "text-sm font-medium transition-opacity duration-200",
                    isCollapsed ? "opacity-0 absolute" : "opacity-100"
                  )}
                >
                  {item.name}
                </span>
              </>
            );

            return (
              <div key={item.name}>
                {isCollapsed ? (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Link
                        href={item.href}
                        className={cn(
                          "flex items-center gap-3 rounded-lg px-3 py-3 justify-center transition-colors",
                          isActive
                            ? "bg-primary text-primary-foreground"
                            : "hover:bg-muted"
                        )}
                      >
                        {linkContent}
                      </Link>
                    </TooltipTrigger>
                    <TooltipContent side="right">
                      <p>{item.name}</p>
                    </TooltipContent>
                  </Tooltip>
                ) : (
                  <Link
                    href={item.href}
                    className={cn(
                      "flex items-center gap-3 rounded-lg px-3 py-3 transition-colors",
                      isActive
                        ? "bg-primary text-primary-foreground"
                        : "hover:bg-muted"
                    )}
                  >
                    {linkContent}
                  </Link>
                )}
              </div>
            );
          })}
        </nav>

        {/* User Profile (at bottom) */}
        <div className="p-2 border-t">
          <div className="flex items-center gap-3 p-2 rounded-lg">
            <div className="h-10 w-10 rounded-full bg-muted border flex items-center justify-center flex-shrink-0">
              <User className="h-5 w-5 text-muted-foreground" />
            </div>
            <div
              className={cn(
                "flex flex-col transition-opacity duration-200",
                isCollapsed ? "opacity-0" : "opacity-100"
              )}
            >
              <span className="text-sm font-medium text-foreground">
                Vivek Vasani
              </span>
              <span className="text-xs text-muted-foreground">Developer</span>
            </div>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}
