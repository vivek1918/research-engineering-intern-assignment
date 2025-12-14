'use client';

import React, { useState, useRef, useEffect } from 'react';
import { format } from 'date-fns';
import { DayPicker } from 'react-day-picker';
import 'react-day-picker/dist/style.css'; // Import default styles
import { Calendar as CalendarIcon } from 'lucide-react';

export function CustomDateRangePicker({ date, setDate, className }) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef(null);

  // Close the popover if the user clicks outside of it
  useEffect(() => {
    function handleClickOutside(event) {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [containerRef]);

  const buttonText =
    date?.from && date?.to
      ? `${format(date.from, 'LLL dd, y')} - ${format(date.to, 'LLL dd, y')}`
      : "Select a date range";

  return (
    <div className="relative" ref={containerRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`w-full sm:w-[300px] justify-start text-left font-normal flex items-center gap-2 px-3 py-2 border border-input bg-background rounded-md text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 ${
          !date && "text-muted-foreground"
        } ${className}`}
      >
        <CalendarIcon className="h-4 w-4" />
        {buttonText}
      </button>

      {isOpen && (
        <div className="absolute z-10 top-full mt-2 bg-card p-2 border rounded-md shadow-lg">
          <DayPicker
            mode="range"
            selected={date}
            onSelect={(range) => {
              setDate(range);
              // Optional: close picker after a range is selected
              if (range?.from && range?.to) {
                setIsOpen(false);
              }
            }}
            numberOfMonths={2}
          />
        </div>
      )}
    </div>
  );
}