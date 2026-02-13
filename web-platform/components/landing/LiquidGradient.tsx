'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

export default function LiquidGradient() {
  const [visible, setVisible] = useState(false);
  const [gradient, setGradient] = useState('transparent');
  const positionRef = useRef({ x: 0, y: 0 });
  const currentRef = useRef({ x: 0, y: 0 });
  const elementRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number | undefined>(undefined);

  const calculateGradient = useCallback((clientX: number, clientY: number) => {
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;

    // Normalize position (0-1)
    const normalizedX = clientX / windowWidth;
    const normalizedY = clientY / windowHeight;

    // Calculate RGB values based on position
    // Left = Orange (#FF6B35), Right = Blue (#00A8E8)
    const r = Math.floor(255 * (1 - normalizedX) + 107 * normalizedX);
    const g = Math.floor(107 * (1 - normalizedX) + 168 * normalizedX);
    const b = Math.floor(53 * (1 - normalizedX) + 232 * normalizedX);

    // Add purple tint based on Y position
    const purpleInfluence = normalizedY * 0.3;
    const finalR = Math.floor(r + (150 - r) * purpleInfluence);
    const finalG = Math.floor(g + (50 - g) * purpleInfluence);
    const finalB = Math.floor(b + (200 - b) * purpleInfluence);

    return `radial-gradient(
      circle at center,
      rgba(${finalR}, ${finalG}, ${finalB}, 0.25) 0%,
      rgba(${finalR}, ${finalG}, ${finalB}, 0.15) 30%,
      transparent 70%
    )`;
  }, []);

  const animate = useCallback(() => {
    const element = elementRef.current;
    if (!element) {
      animationRef.current = requestAnimationFrame(animate);
      return;
    }

    // Smooth interpolation
    currentRef.current.x += (positionRef.current.x - currentRef.current.x) * 0.1;
    currentRef.current.y += (positionRef.current.y - currentRef.current.y) * 0.1;

    element.style.left = `${currentRef.current.x}px`;
    element.style.top = `${currentRef.current.y}px`;

    animationRef.current = requestAnimationFrame(animate);
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      positionRef.current = { x: e.clientX, y: e.clientY };
      setVisible(true);
      setGradient(calculateGradient(e.clientX, e.clientY));
    };

    const handleMouseLeave = () => {
      setVisible(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseleave', handleMouseLeave);

    // Start animation loop
    animate();

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseleave', handleMouseLeave);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [animate, calculateGradient]);

  return (
    <div
      ref={elementRef}
      className="fixed w-[250px] h-[250px] rounded-full pointer-events-none z-[2] -translate-x-1/2 -translate-y-1/2 blur-[60px] mix-blend-screen transition-opacity duration-500"
      style={{
        opacity: visible ? 1 : 0,
        background: gradient,
      }}
      aria-hidden="true"
    />
  );
}
