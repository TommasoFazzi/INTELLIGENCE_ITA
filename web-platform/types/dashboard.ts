// Dashboard Stats API Response Types

export interface SourceCount {
  source: string;
  count: number;
}

export interface EntityMention {
  name: string;
  type: string;
  mentions: number;
}

export interface DateRange {
  first: string | null;
  last: string | null;
}

export interface OverviewStats {
  total_articles: number;
  total_entities: number;
  total_reports: number;
  geocoded_entities: number;
  coverage_percentage: number;
}

export interface ArticleStats {
  by_category: Record<string, number>;
  by_source: SourceCount[];
  recent_7d: number;
  date_range: DateRange;
}

export interface EntityStats {
  by_type: Record<string, number>;
  top_mentioned: EntityMention[];
}

export interface QualityStats {
  reports_reviewed: number;
  average_rating: number | null;
  pending_review: number;
}

export interface DashboardStats {
  overview: OverviewStats;
  articles: ArticleStats;
  entities: EntityStats;
  quality: QualityStats;
}

export interface DashboardStatsResponse {
  success: boolean;
  data: DashboardStats;
  error: string | null;
  generated_at: string;
}

// Reports API Response Types

export type ReportStatus = 'draft' | 'reviewed' | 'approved';
export type ReportType = 'daily' | 'weekly';

export interface ReportListItem {
  id: number;
  report_date: string;
  report_type: ReportType;
  status: ReportStatus;
  title: string | null;
  category: string | null;
  executive_summary: string | null;
  article_count: number;
  generated_at: string | null;
  reviewed_at: string | null;
  reviewer: string | null;
}

export interface Pagination {
  page: number;
  per_page: number;
  total: number;
  pages: number;
}

export interface ReportsFilters {
  status: ReportStatus | null;
  report_type: ReportType | null;
  date_from: string | null;
  date_to: string | null;
}

export interface ReportsData {
  reports: ReportListItem[];
  pagination: Pagination;
  filters_applied: ReportsFilters;
}

export interface ReportsResponse {
  success: boolean;
  data: ReportsData;
  error: string | null;
  generated_at: string;
}

// Report Detail Types

export interface ReportSource {
  article_id: number;
  title: string;
  link: string;
  relevance_score: number | null;
}

export interface ReportFeedback {
  section: string;
  rating: number | null;
  comment: string | null;
}

export interface ReportSection {
  category: string;
  content: string;
  entities: string[];
}

export interface ReportContent {
  title: string;
  executive_summary: string;
  full_text: string;
  sections: ReportSection[];
}

export interface ReportMetadata {
  processing_time_ms: number | null;
  token_count: number | null;
}

export interface ReportDetail {
  id: number;
  report_date: string;
  report_type: ReportType;
  status: ReportStatus;
  model_used: string | null;
  content: ReportContent;
  sources: ReportSource[];
  feedback: ReportFeedback[];
  metadata: ReportMetadata;
}

export interface ReportDetailResponse {
  success: boolean;
  data: ReportDetail;
  generated_at: string;
}

// Error types
export interface ApiError extends Error {
  status?: number;
  isOffline: boolean;
}
