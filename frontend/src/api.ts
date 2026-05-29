const BASE = '/api';

// ── 타입 정의 ──────────────────────────────────────────────────────────

export interface Worker {
  id: number;
  count: number;
  created_at: string;
}

export type Maker = Worker;

export interface IncidentLog {
  id: number;
  worker_id: number;
  maker_id: number;
  incident_type: string;
  snapshot_path: string;
  status: string;
  created_at: string;
}

export interface WorkerIncidentSummary {
  worker_id: number;
  total: number;
  warning: number;
  danger: number;
}

export interface IncidentLogSummary {
  target_date: string;
  total: number;
  warning: number;
  danger: number;
  workers: WorkerIncidentSummary[];
}

export interface Report {
  id: number;
  contents: string;
  date: string;
  created_at: string;
}

export interface ReportSummary {
  id: number;
  date: string;
  created_at: string;
  contents_length: number;
}

export interface JobSubmit {
  job_id: string;
  job_type: string;
  status: string;
  status_url?: string | null;
}

export interface JobStatus {
  job_id: string;
  job_type: string;
  status: 'queued' | 'running' | 'done' | 'failed' | string;
  payload?: Record<string, unknown> | null;
  result?: {
    report_id?: number;
    date?: string;
    created_at?: string;
  } | null;
  error?: string | null;
  created_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  queued_ms?: number | null;
  runtime_ms?: number | null;
  total_ms?: number | null;
}

// ── API 호출 함수 ───────────────────────────────────────────────────────

export async function fetchMakers(): Promise<Maker[]> {
  const res = await fetch(`${BASE}/makers`);
  if (!res.ok) throw new Error('makers 조회 실패');
  return res.json();
}

export async function fetchWorkers(): Promise<Worker[]> {
  const res = await fetch(`${BASE}/workers`);
  if (!res.ok) throw new Error('workers 조회 실패');
  return res.json();
}

export async function fetchIncidentLogs(): Promise<IncidentLog[]> {
  const res = await fetch(`${BASE}/incident-logs`);
  if (!res.ok) throw new Error('incident-logs 조회 실패');
  return res.json();
}

export async function fetchIncidentSummary(targetDate: string): Promise<IncidentLogSummary> {
  const res = await fetch(`${BASE}/incident-logs/summary?target_date=${encodeURIComponent(targetDate)}`);
  if (!res.ok) throw new Error('incident-logs summary 조회 실패');
  return res.json();
}

export async function fetchReports(): Promise<Report[]> {
  const res = await fetch(`${BASE}/reports`);
  if (!res.ok) throw new Error('reports 조회 실패');
  return res.json();
}

export async function fetchReportSummaries(limit = 50, offset = 0): Promise<ReportSummary[]> {
  const res = await fetch(`${BASE}/reports/summary?limit=${limit}&offset=${offset}`);
  if (!res.ok) throw new Error('reports summary 조회 실패');
  return res.json();
}

export async function fetchReport(reportId: number): Promise<Report> {
  const res = await fetch(`${BASE}/reports/${reportId}`);
  if (!res.ok) throw new Error('report detail 조회 실패');
  return res.json();
}

export async function sendAlert(workerId: string, direction: string): Promise<{ status: string }> {
  const res = await fetch(`${BASE}/send-alert?worker_id=${encodeURIComponent(workerId)}&direction=${encodeURIComponent(direction)}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('send-alert 실패');
  return res.json();
}

export async function generateReport(targetDate?: string, requestId?: string): Promise<Report> {
  const query = targetDate ? `?target_date=${targetDate}` : '';
  const res = await fetch(`${BASE}/reports/generate${query}`, {
    method: 'POST',
    headers: requestId ? { 'X-Request-ID': requestId } : undefined,
  });
  if (!res.ok) {
    if (res.status === 404) throw new Error('NO_DATA');
    throw new Error('SERVER_ERROR');
  }
  return res.json();
}

export async function generateReportAsync(targetDate?: string): Promise<JobSubmit> {
  const query = targetDate ? `?target_date=${encodeURIComponent(targetDate)}` : '';
  const res = await fetch(`${BASE}/reports/generate-async${query}`, {
    method: 'POST',
  });
  if (!res.ok) {
    if (res.status === 404) throw new Error('NO_DATA');
    throw new Error('SERVER_ERROR');
  }
  return res.json();
}

export async function fetchJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error('job status 조회 실패');
  return res.json();
}
