// src/app/layout.jsx
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
// FIX: Corrected the import path to include the 'ui' folder
import Sidebar from "@/app/components/ui/sidebar"; 

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata = {
  title: "SocialPulse AI",
  description: "Your AI-powered data intelligence dashboard.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="h-full">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased h-full bg-gradient-to-br from-slate-50 to-blue-50`}
      >
        <div className="flex h-full">
          <Sidebar />
+          <main className="flex-1 flex flex-col overflow-y-auto p-8"> {/* Added p-8 for space around content */}            {children}
          </main>
        </div>
      </body>
    </html>
  );
}