const BASE = '/api';

// ── 타입 정의 ──────────────────────────────────────────────────────────

export interface Maker {
  id: number;
  count: number;
  created_at: string;
}

export interface IncidentLog {
  id: number;
  maker_id: number;
  incident_type: string;
  snapshot_path: string;
  status: string;
  created_at: string;
}

export interface Report {
  id: number;
  contents: string;
  date: string;
  created_at: string;
}

// ── API 호출 함수 ───────────────────────────────────────────────────────

export async function fetchMakers(): Promise<Maker[]> {
  const res = await fetch(`${BASE}/makers`);
  if (!res.ok) throw new Error('makers 조회 실패');
  return res.json();
}

export async function fetchIncidentLogs(): Promise<IncidentLog[]> {
  const res = await fetch(`${BASE}/incident-logs`);
  if (!res.ok) throw new Error('incident-logs 조회 실패');
  return res.json();
}

export async function fetchReports(): Promise<Report[]> {
  const res = await fetch(`${BASE}/reports`);
  if (!res.ok) throw new Error('reports 조회 실패');
  return res.json();
}

export async function sendAlert(makerId: string, direction: string): Promise<{ status: string }> {
  const res = await fetch(`${BASE}/send-alert?maker_id=${encodeURIComponent(makerId)}&direction=${encodeURIComponent(direction)}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('send-alert 실패');
  return res.json();
}

export async function generateReport(targetDate?: string): Promise<Report> {
  const query = targetDate ? `?target_date=${targetDate}` : '';
  const res = await fetch(`${BASE}/reports/generate${query}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('report 생성 실패');
  return res.json();
}
