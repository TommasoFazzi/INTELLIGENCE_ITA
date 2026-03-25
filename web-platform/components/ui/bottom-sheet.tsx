'use client';

import { useEffect, useRef } from 'react';
import { X } from 'lucide-react';

interface BottomSheetProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  /** Tailwind height class for the sheet (default: 'h-[78vh]') */
  heightClass?: string;
  /** Additional className on the sheet panel */
  className?: string;
}

/**
 * Mobile-only bottom sheet overlay.
 * On md+ screens nothing is rendered — use a side panel instead.
 * Features: backdrop tap to close, drag-handle swipe-down to dismiss.
 */
export function BottomSheet({
  open,
  onClose,
  title,
  children,
  heightClass = 'h-[78vh]',
  className = '',
}: BottomSheetProps) {
  const sheetRef = useRef<HTMLDivElement>(null);
  const startYRef = useRef<number>(0);
  const isDraggingRef = useRef(false);

  // Lock body scroll while open
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = prev; };
  }, [open]);

  if (!open) return null;

  const handleHandleTouchStart = (e: React.TouchEvent) => {
    startYRef.current = e.touches[0].clientY;
    isDraggingRef.current = true;
    if (sheetRef.current) sheetRef.current.style.transition = 'none';
  };

  const handleHandleTouchMove = (e: React.TouchEvent) => {
    if (!isDraggingRef.current || !sheetRef.current) return;
    const dy = e.touches[0].clientY - startYRef.current;
    if (dy > 0) {
      sheetRef.current.style.transform = `translateY(${dy}px)`;
    }
  };

  const handleHandleTouchEnd = (e: React.TouchEvent) => {
    if (!isDraggingRef.current || !sheetRef.current) return;
    isDraggingRef.current = false;
    sheetRef.current.style.transition = '';
    const dy = e.changedTouches[0].clientY - startYRef.current;
    if (dy > 80) {
      onClose();
    } else {
      sheetRef.current.style.transform = '';
    }
  };

  return (
    /* Only visible on mobile — hidden on md+ (desktop uses its own side panel) */
    <div className="fixed inset-0 z-50 md:hidden">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Sheet */}
      <div
        ref={sheetRef}
        role="dialog"
        aria-modal="true"
        className={`absolute bottom-0 left-0 right-0 flex flex-col bg-gray-900 border-t border-white/10 rounded-t-2xl shadow-2xl transition-transform duration-300 ${heightClass} ${className}`}
      >
        {/* Drag handle */}
        <div
          className="flex-shrink-0 flex flex-col items-center pt-3 pb-2 touch-none cursor-grab active:cursor-grabbing"
          onTouchStart={handleHandleTouchStart}
          onTouchMove={handleHandleTouchMove}
          onTouchEnd={handleHandleTouchEnd}
        >
          <div className="w-10 h-1 rounded-full bg-white/20" />

          {title && (
            <div className="flex items-center justify-between w-full px-4 mt-3">
              <span className="text-sm font-semibold text-white/80">{title}</span>
              <button
                type="button"
                onClick={onClose}
                className="p-1.5 text-gray-400 active:text-white transition-colors"
                aria-label="Close"
              >
                <X size={20} />
              </button>
            </div>
          )}
        </div>

        {/* Scrollable content + home indicator safe area */}
        <div className="flex-1 overflow-y-auto bottom-sheet-safe">
          {children}
        </div>
      </div>
    </div>
  );
}
