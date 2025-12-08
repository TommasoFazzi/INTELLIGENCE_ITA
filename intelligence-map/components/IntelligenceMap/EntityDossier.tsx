'use client';

import { X, MapPin, Calendar, FileText, ExternalLink } from 'lucide-react';

interface Article {
  id: number;
  title: string;
  link: string;
  published_date: string;
  source: string;
}

interface EntityData {
  id: number;
  name: string;
  entity_type: string;
  latitude: number;
  longitude: number;
  mention_count: number;
  first_seen: string;
  last_seen: string;
  metadata: Record<string, any>;
  related_articles: Article[];
}

interface EntityDossierProps {
  entity: EntityData | null;
  onClose: () => void;
}

export default function EntityDossier({ entity, onClose }: EntityDossierProps) {
  if (!entity) return null;

  const formatDate = (dateString: string) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('it-IT', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const getEntityTypeLabel = (type: string) => {
    const labels: Record<string, string> = {
      'GPE': 'Geopolitical Entity',
      'LOC': 'Location',
      'FAC': 'Facility',
      'PERSON': 'Person',
      'ORG': 'Organization'
    };
    return labels[type] || type;
  };

  const getEntityTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      'GPE': 'text-cyan-400',
      'LOC': 'text-green-400',
      'FAC': 'text-orange-400',
      'PERSON': 'text-purple-400',
      'ORG': 'text-yellow-400'
    };
    return colors[type] || 'text-gray-400';
  };

  return (
    <div className="fixed right-4 top-4 bottom-4 w-[450px] bg-gray-900/95 backdrop-blur-sm border-2 border-cyan-500/30 shadow-2xl z-50 flex flex-col">
      {/* Header */}
      <div className="bg-gradient-to-r from-cyan-900/50 to-gray-900/50 border-b-2 border-cyan-500/30 p-4">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
              <span className="text-xs font-mono text-cyan-400 uppercase tracking-wider">
                Entity Dossier
              </span>
            </div>
            <h2 className="text-2xl font-bold text-white mb-1">{entity.name}</h2>
            <div className="flex items-center gap-2">
              <span className={`text-sm font-mono ${getEntityTypeColor(entity.entity_type)}`}>
                {getEntityTypeLabel(entity.entity_type)}
              </span>
              <span className="text-gray-500">•</span>
              <span className="text-sm text-gray-400 font-mono">
                ID: {entity.id.toString().padStart(6, '0')}
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors p-1"
            aria-label="Close dossier"
          >
            <X size={24} />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Intelligence Summary */}
        <div className="border border-cyan-500/20 bg-gray-800/50 p-4 rounded">
          <h3 className="text-sm font-mono text-cyan-400 uppercase mb-3 flex items-center gap-2">
            <MapPin size={14} />
            Intelligence Summary
          </h3>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <div className="text-gray-500 text-xs mb-1">Coordinates</div>
              <div className="font-mono text-white">
                {entity.latitude.toFixed(4)}°N<br />
                {entity.longitude.toFixed(4)}°E
              </div>
            </div>
            <div>
              <div className="text-gray-500 text-xs mb-1">Mention Count</div>
              <div className="font-mono text-cyan-400 text-xl font-bold">
                {entity.mention_count}
              </div>
            </div>
            <div>
              <div className="text-gray-500 text-xs mb-1">First Seen</div>
              <div className="font-mono text-white text-xs">
                {formatDate(entity.first_seen)}
              </div>
            </div>
            <div>
              <div className="text-gray-500 text-xs mb-1">Last Seen</div>
              <div className="font-mono text-white text-xs">
                {formatDate(entity.last_seen)}
              </div>
            </div>
          </div>
        </div>

        {/* Related Articles */}
        <div className="border border-cyan-500/20 bg-gray-800/50 p-4 rounded">
          <h3 className="text-sm font-mono text-cyan-400 uppercase mb-3 flex items-center gap-2">
            <FileText size={14} />
            Related Intelligence ({entity.related_articles?.length || 0})
          </h3>

          {entity.related_articles && entity.related_articles.length > 0 ? (
            <div className="space-y-3 max-h-[400px] overflow-y-auto">
              {entity.related_articles.map((article) => (
                <div
                  key={article.id}
                  className="border-l-2 border-cyan-500/30 pl-3 py-2 hover:border-cyan-500 transition-colors group"
                >
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <div className="text-xs font-mono text-gray-500 flex items-center gap-2">
                      <Calendar size={12} />
                      {formatDate(article.published_date)}
                    </div>
                    <a
                      href={article.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-cyan-400 hover:text-cyan-300 transition-colors"
                      aria-label="Open article"
                    >
                      <ExternalLink size={14} />
                    </a>
                  </div>
                  <h4 className="text-sm text-white group-hover:text-cyan-400 transition-colors leading-snug mb-1">
                    {article.title}
                  </h4>
                  <div className="text-xs text-gray-500 font-mono">
                    Source: {article.source}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-sm text-gray-500 text-center py-4">
              No related articles found
            </div>
          )}
        </div>

        {/* Metadata (if available) */}
        {entity.metadata && Object.keys(entity.metadata).length > 0 && (
          <div className="border border-cyan-500/20 bg-gray-800/50 p-4 rounded">
            <h3 className="text-sm font-mono text-cyan-400 uppercase mb-3">
              Additional Metadata
            </h3>
            <pre className="text-xs text-gray-400 font-mono overflow-x-auto">
              {JSON.stringify(entity.metadata, null, 2)}
            </pre>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="border-t-2 border-cyan-500/30 bg-gray-900/50 px-4 py-3">
        <div className="text-xs text-gray-500 font-mono text-center">
          CLASSIFIED • INTELLIGENCE_ITA • {new Date().toISOString().split('T')[0]}
        </div>
      </div>
    </div>
  );
}
