'use client';

import { useCallback, useRef, useState, useMemo } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { useGraphNetwork } from '@/hooks/useStories';
import StorylineDossier from './StorylineDossier';
import type { NarrativeStatus } from '@/types/stories';

const STATUS_COLORS: Record<NarrativeStatus, string> = {
  emerging: '#FF6B35',
  active: '#00A8E8',
  stabilized: '#666666',
};

interface GraphNode {
  id: number;
  title: string;
  narrative_status: NarrativeStatus;
  momentum_score: number;
  article_count: number;
  category: string | null;
  x?: number;
  y?: number;
}

interface GraphLink {
  source: number | GraphNode;
  target: number | GraphNode;
  weight: number;
  relation_type: string;
}

export default function StorylineGraph() {
  const graphRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const { graph, isLoading, error, refresh } = useGraphNetwork();
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);

  // Transform API data for react-force-graph
  const graphData = useMemo(() => {
    if (!graph) return { nodes: [], links: [] };

    const nodes: GraphNode[] = graph.nodes.map((n) => ({
      id: n.id,
      title: n.title,
      narrative_status: n.narrative_status as NarrativeStatus,
      momentum_score: n.momentum_score,
      article_count: n.article_count,
      category: n.category,
    }));

    const nodeIds = new Set(nodes.map((n) => n.id));
    const links: GraphLink[] = graph.links
      .filter((l) => nodeIds.has(l.source) && nodeIds.has(l.target))
      .map((l) => ({
        source: l.source,
        target: l.target,
        weight: l.weight,
        relation_type: l.relation_type,
      }));

    return { nodes, links };
  }, [graph]);

  // Node rendering
  const paintNode = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const { x, y, title, momentum_score, narrative_status } = node as GraphNode;
      if (x === undefined || y === undefined) return;

      const isSelected = node.id === selectedId;
      const isHovered = hoveredNode?.id === node.id;
      const color = STATUS_COLORS[narrative_status] || STATUS_COLORS.active;

      // Node radius based on momentum (min 4, max 16)
      const radius = 4 + momentum_score * 12;

      // Glow effect for selected/hovered
      if (isSelected || isHovered) {
        ctx.beginPath();
        ctx.arc(x, y, radius + 4, 0, 2 * Math.PI);
        ctx.fillStyle = `${color}33`;
        ctx.fill();
      }

      // Main circle
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = isSelected ? '#FFFFFF' : color;
      ctx.fill();

      // Border
      ctx.strokeStyle = isSelected ? color : `${color}88`;
      ctx.lineWidth = isSelected ? 2 : 1;
      ctx.stroke();

      // Label (only show when zoomed in enough or for high-momentum nodes)
      if (globalScale > 1.5 || momentum_score > 0.7 || isHovered || isSelected) {
        const label = title.length > 30 ? title.slice(0, 30) + '...' : title;
        const fontSize = Math.max(10 / globalScale, 3);
        ctx.font = `${fontSize}px monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';

        // Text background
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = 'rgba(10, 22, 40, 0.85)';
        ctx.fillRect(
          x - textWidth / 2 - 2,
          y + radius + 2,
          textWidth + 4,
          fontSize + 4
        );

        // Text
        ctx.fillStyle = isSelected ? '#FFFFFF' : '#CCCCCC';
        ctx.fillText(label, x, y + radius + 4);
      }
    },
    [selectedId, hoveredNode]
  );

  // Link rendering
  const paintLink = useCallback(
    (link: any, ctx: CanvasRenderingContext2D) => {
      const { source, target, weight } = link;
      if (!source.x || !target.x) return;

      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.strokeStyle = `rgba(100, 100, 100, ${0.2 + weight * 0.6})`;
      ctx.lineWidth = 0.5 + weight * 2.5;
      ctx.stroke();
    },
    []
  );

  const handleNodeClick = useCallback((node: any) => {
    setSelectedId((prev) => (prev === node.id ? null : node.id));
  }, []);

  const handleNavigate = useCallback((id: number) => {
    setSelectedId(id);
    // Center on the node
    if (graphRef.current) {
      const node = graphData.nodes.find((n) => n.id === id);
      if (node && node.x !== undefined && node.y !== undefined) {
        graphRef.current.centerAt(node.x, node.y, 500);
        graphRef.current.zoom(3, 500);
      }
    }
  }, [graphData.nodes]);

  return (
    <div ref={containerRef} className="relative w-full h-screen bg-[#0A1628] overflow-hidden">
      {/* Force Graph */}
      <ForceGraph2D
        ref={graphRef}
        graphData={graphData}
        nodeId="id"
        nodeCanvasObject={paintNode}
        nodePointerAreaPaint={(node: any, color, ctx) => {
          const radius = 4 + (node.momentum_score || 0.5) * 12;
          ctx.beginPath();
          ctx.arc(node.x!, node.y!, radius + 4, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
        }}
        linkCanvasObject={paintLink}
        onNodeClick={handleNodeClick}
        onNodeHover={(node: any) => setHoveredNode(node || null)}
        backgroundColor="#0A1628"
        cooldownTicks={100}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
        linkDirectionalParticles={0}
        enableNodeDrag={true}
        enableZoomInteraction={true}
        enablePanInteraction={true}
      />

      {/* HUD Overlay - Top Left */}
      <div className="absolute top-4 left-4 pointer-events-none">
        <div className="bg-[#0A1628]/80 backdrop-blur-sm border border-[#FF6B35]/30 rounded px-4 py-3">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 bg-[#FF6B35] rounded-full animate-pulse" />
            <span className="text-[#FF6B35] font-mono text-sm font-bold tracking-wider">
              NARRATIVE GRAPH
            </span>
          </div>
          {graph?.stats && (
            <div className="space-y-1 text-xs font-mono text-gray-400">
              <div>NODES: <span className="text-white">{graph.stats.total_nodes}</span></div>
              <div>EDGES: <span className="text-white">{graph.stats.total_edges}</span></div>
              <div>AVG MOMENTUM: <span className="text-white">{graph.stats.avg_momentum.toFixed(2)}</span></div>
            </div>
          )}
        </div>
      </div>

      {/* Legend - Top Right */}
      <div className="absolute top-4 right-4 pointer-events-none">
        {!selectedId && (
          <div className="bg-[#0A1628]/80 backdrop-blur-sm border border-white/10 rounded px-4 py-3">
            <div className="text-xs font-mono text-gray-500 mb-2 uppercase">Status Legend</div>
            <div className="space-y-1.5">
              {Object.entries(STATUS_COLORS).map(([status, color]) => (
                <div key={status} className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: color }}
                  />
                  <span className="text-xs font-mono text-gray-300 capitalize">{status}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Hovered node tooltip */}
      {hoveredNode && !selectedId && (
        <div className="absolute bottom-4 left-4 pointer-events-none">
          <div className="bg-[#0A1628]/90 backdrop-blur-sm border border-[#FF6B35]/30 rounded px-4 py-3 max-w-sm">
            <div className="text-white font-mono text-sm font-bold mb-1">
              {hoveredNode.title}
            </div>
            <div className="flex items-center gap-3 text-xs font-mono text-gray-400">
              <span>Momentum: <span className="text-[#FF6B35]">{hoveredNode.momentum_score.toFixed(2)}</span></span>
              <span>Articles: <span className="text-white">{hoveredNode.article_count}</span></span>
              {hoveredNode.category && (
                <span className="text-[#00A8E8]">{hoveredNode.category}</span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Loading overlay */}
      {isLoading && !graph && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#0A1628]/80">
          <div className="text-center">
            <div className="w-12 h-12 border-2 border-[#FF6B35] border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <div className="text-[#FF6B35] font-mono text-sm">Loading graph data...</div>
          </div>
        </div>
      )}

      {/* Error state */}
      {error && !graph && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#0A1628]/80">
          <div className="text-center max-w-md">
            <div className="text-red-400 font-mono text-lg mb-2">Connection Error</div>
            <div className="text-gray-400 font-mono text-sm mb-4">
              Unable to load narrative graph data. Make sure the API server is running.
            </div>
            <button
              onClick={() => refresh()}
              className="px-4 py-2 bg-[#FF6B35]/20 border border-[#FF6B35]/40 text-[#FF6B35] font-mono text-sm rounded hover:bg-[#FF6B35]/30 transition-colors"
            >
              RETRY
            </button>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!isLoading && !error && graph && graph.nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className="text-gray-500 font-mono text-lg mb-2">No Active Storylines</div>
            <div className="text-gray-600 font-mono text-sm">
              Run the narrative pipeline to generate storylines.
            </div>
          </div>
        </div>
      )}

      {/* Storyline Dossier Panel */}
      <StorylineDossier
        storylineId={selectedId}
        onClose={() => setSelectedId(null)}
        onNavigate={handleNavigate}
      />

      {/* Corner brackets */}
      <div className="absolute top-2 left-2 w-8 h-8 border-l-2 border-t-2 border-[#FF6B35]/30 pointer-events-none" />
      <div className="absolute top-2 right-2 w-8 h-8 border-r-2 border-t-2 border-[#FF6B35]/30 pointer-events-none" />
      <div className="absolute bottom-2 left-2 w-8 h-8 border-l-2 border-b-2 border-[#FF6B35]/30 pointer-events-none" />
      <div className="absolute bottom-2 right-2 w-8 h-8 border-r-2 border-b-2 border-[#FF6B35]/30 pointer-events-none" />
    </div>
  );
}
