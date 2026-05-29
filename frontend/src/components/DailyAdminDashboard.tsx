import { useState, useRef, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import {
  LayoutDashboard,
  Calendar,
  Download,
  AlertTriangle,
  Activity,
  User,
  ShieldCheck,
  Zap,
  Radio,
  Eye,
  ChevronDown,
  Sparkles,
  ClipboardList,
  NotebookPen,
  FileWarning,
  ScrollText
} from 'lucide-react';
import { toPng } from 'html-to-image';
import jsPDF from 'jspdf';
import {
  fetchIncidentSummary,
  fetchJobStatus,
  fetchReport,
  fetchReportSummaries,
  fetchWorkers,
  generateReportAsync,
  sendAlert,
} from '../api';
import type { IncidentLogSummary, ReportSummary, Worker } from '../api';

// === Mock Data ===
const USE_MOCK_DATA = false;

const MOCK_INCIDENTS = [
  {
    time: '2026-04-17 14:32:05',
    worker: '작업자 1 - 김현수',
    event: '크레인 위험 구역 접근 감지',
    action: '안전모 진동 경고 발송 완료',
    severity: '높음',
    severityColor: 'bg-red-50 text-red-600 border-red-200',
  },
  {
    time: '2026-04-17 10:15:22',
    worker: '작업자 3 - 박상범',
    event: '크레인 파단음 구역 이탈 지연',
    action: '현장 스피커 대피 방송 송출 (2회)',
    severity: '위험',
    severityColor: 'bg-rose-50 text-rose-600 border-rose-200',
  },
  {
    time: '2026-04-17 08:45:10',
    worker: '작업자 2 - 이철수',
    event: '지게차 이동 경로 사각지대 침범',
    action: '스마트 조끼 경고(우측 진동)',
    severity: '주의',
    severityColor: 'bg-amber-50 text-amber-600 border-amber-200',
  }
];

const lineChartData = [
  { time: '09:00', count: 2 },
  { time: '10:00', count: 5 },
  { time: '11:00', count: 3 },
  { time: '12:00', count: 0 },
  { time: '13:00', count: 6 },
  { time: '14:00', count: 8 },
  { time: '15:00', count: 4 },
  { time: '16:00', count: 2 },
  { time: '17:00', count: 1 },
  { time: '18:00', count: 0 },
];

const initialWorkers: Worker[] = [
  { id: 1, count: 0, created_at: '' },
  { id: 2, count: 0, created_at: '' },
];

const REPORT_SUMMARY_LIMIT = 200;
const REPORT_POLL_INTERVAL_MS = 1_000;
const REPORT_POLL_TIMEOUT_MS = 180_000;

type ReportGenerationTiming = {
  requestId: string;
  targetDate: string;
  startedAt: string;
  clickToResponseMs: number;
  responseToRenderMs: number;
  clickToRenderMs: number;
  reportId?: number;
  htmlBytes: number;
};

declare global {
  interface Window {
    __pobigaReportMetrics?: ReportGenerationTiming[];
  }
}

function makeRequestId(prefix: string) {
  const randomId = window.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2);
  return `${prefix}-${Date.now()}-${randomId}`;
}

function getLocalDateString() {
  const now = new Date();
  const local = new Date(now.getTime() - now.getTimezoneOffset() * 60_000);
  return local.toISOString().slice(0, 10);
}

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export default function DailyAdminDashboard() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'records' | 'vibration'>('dashboard');

  // === 전역(Global) 상태 및 헤더(Top Panel) 연동 ===
  const [selectedDate, setSelectedDate] = useState(getLocalDateString());

  // 아코디언 열림/닫힘 관리를 위한 상태
  const [expandedProfileId, setExpandedProfileId] = useState<number | null>(null);

  const [workers, setWorkers] = useState<Worker[]>(initialWorkers);
  const [incidentSummary, setIncidentSummary] = useState<IncidentLogSummary | null>(null);
  const [summaryStatus, setSummaryStatus] = useState<'loading' | 'ok' | 'error'>('loading');
  const dailyCountByWorker = new Map(
    incidentSummary?.workers.map((worker) => [worker.worker_id, worker]) ?? [],
  );
  const profiles = workers.map((worker) => {
    const daily = dailyCountByWorker.get(worker.id);
    const dailyCount = daily?.total ?? 0;
    const score = Math.max(0, 100 - dailyCount * 8);
    return {
      id: worker.id,
      name: `작업자 ${worker.id}`,
      score,
      count: dailyCount,
      allTimeCount: worker.count,
      warning: daily?.warning ?? 0,
      danger: daily?.danger ?? 0,
      violations: dailyCount > 0
        ? [
            `${selectedDate} 위험 알림 ${dailyCount}회`,
            `Warning ${daily?.warning ?? 0}회 / Danger ${daily?.danger ?? 0}회`,
            `전체 누적 ${worker.count}회`,
          ]
        : [],
    };
  });
  const totalRiskCount = incidentSummary?.total ?? 0;
  const totalWarningCount = incidentSummary?.warning ?? 0;
  const totalDangerCount = incidentSummary?.danger ?? 0;

  // === 일자별 기록 뷰: 서버 리포트 연동 ===
  const [reportSummaries, setReportSummaries] = useState<ReportSummary[]>([]);
  const [reportSummariesLoaded, setReportSummariesLoaded] = useState(false);
  const [reportHtml, setReportHtml] = useState('');
  const [reportStatus, setReportStatus] = useState<'loading' | 'empty' | 'generating' | 'generate_no_data' | 'error' | 'ok'>('loading');
  const pendingReportTimingRef = useRef<{
    requestId: string;
    targetDate: string;
    startedAt: string;
    clickStartMs: number;
    responseEndMs: number;
    reportId?: number;
    htmlBytes: number;
  } | null>(null);

  useEffect(() => {
    fetchWorkers()
      .then((data) => {
        if (data.length > 0) {
          setWorkers(data);
        }
      })
      .catch(() => {
        setWorkers(initialWorkers);
      });

    if (USE_MOCK_DATA) {
      setReportStatus('ok');
      return;
    }
    setReportStatus('loading');
    fetchReportSummaries(REPORT_SUMMARY_LIMIT)
      .then((data) => {
        setReportSummaries(data);
        setReportSummariesLoaded(true);
      })
      .catch(() => {
        setReportSummariesLoaded(true);
        setReportStatus('error');
      });
  }, []);

  useEffect(() => {
    setSummaryStatus('loading');
    fetchIncidentSummary(selectedDate)
      .then((summary) => {
        setIncidentSummary(summary);
        setSummaryStatus('ok');
      })
      .catch(() => {
        setIncidentSummary(null);
        setSummaryStatus('error');
      });
  }, [selectedDate]);

  // 날짜 변경 시 해당 날짜 리포트로 전환
  useEffect(() => {
    if (USE_MOCK_DATA) {
      setReportStatus('ok');
      return;
    }
    if (!reportSummariesLoaded) {
      return;
    }

    const matched = reportSummaries.find((r) => r.date === selectedDate);
    if (!matched) {
      setReportHtml('');
      setReportStatus('empty');
      return;
    }

    let cancelled = false;
    setReportStatus('loading');
    fetchReport(matched.id)
      .then((report) => {
        if (cancelled) return;
        setReportHtml(report.contents);
        setReportStatus('ok');
      })
      .catch(() => {
        if (cancelled) return;
        setReportHtml('');
        setReportStatus('error');
      });

    return () => {
      cancelled = true;
    };
  }, [selectedDate, reportSummaries, reportSummariesLoaded]);

  const handleGenerateReport = async () => {
    const requestId = makeRequestId('report');
    const targetDate = selectedDate;
    const clickStartMs = performance.now();
    const startedAt = new Date().toISOString();
    setReportStatus('generating');
    try {
      const job = await generateReportAsync(targetDate);
      const deadlineMs = performance.now() + REPORT_POLL_TIMEOUT_MS;
      let reportId: number | undefined;

      while (performance.now() < deadlineMs) {
        const status = await fetchJobStatus(job.job_id);
        if (status.status === 'done') {
          reportId = status.result?.report_id;
          break;
        }
        if (status.status === 'failed') {
          throw new Error(status.error || 'SERVER_ERROR');
        }
        await delay(REPORT_POLL_INTERVAL_MS);
      }

      if (!reportId) {
        throw new Error('REPORT_TIMEOUT');
      }

      const newReport = await fetchReport(reportId);
      const responseEndMs = performance.now();
      const summaries = await fetchReportSummaries(REPORT_SUMMARY_LIMIT);
      pendingReportTimingRef.current = {
        requestId,
        targetDate,
        startedAt,
        clickStartMs,
        responseEndMs,
        reportId: newReport.id,
        htmlBytes: new Blob([newReport.contents]).size,
      };
      setReportSummaries(summaries);
      setReportSummariesLoaded(true);
      setReportHtml(newReport.contents);
      setReportStatus('ok');
    } catch (e) {
      const failedAtMs = performance.now();
      console.table({
        requestId,
        targetDate: selectedDate,
        status: 'failed',
        clickToFailureMs: Number((failedAtMs - clickStartMs).toFixed(2)),
      });
      if (e instanceof Error && (e.message === 'NO_DATA' || e.message.includes('No incident logs'))) {
        setReportStatus('generate_no_data');
      } else {
        setReportStatus('error');
      }
    }
  };

  const reportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (reportStatus !== 'ok' || !reportHtml || !pendingReportTimingRef.current) {
      return;
    }

    const pending = pendingReportTimingRef.current;
    pendingReportTimingRef.current = null;
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const renderEndMs = performance.now();
        const timing: ReportGenerationTiming = {
          requestId: pending.requestId,
          targetDate: pending.targetDate,
          startedAt: pending.startedAt,
          clickToResponseMs: Number((pending.responseEndMs - pending.clickStartMs).toFixed(2)),
          responseToRenderMs: Number((renderEndMs - pending.responseEndMs).toFixed(2)),
          clickToRenderMs: Number((renderEndMs - pending.clickStartMs).toFixed(2)),
          reportId: pending.reportId,
          htmlBytes: pending.htmlBytes,
        };
        window.__pobigaReportMetrics = [...(window.__pobigaReportMetrics ?? []), timing];
        localStorage.setItem(
          'pobigaReportMetrics',
          JSON.stringify(window.__pobigaReportMetrics),
        );
        console.group('[Pobiga] report generation timing');
        console.table(timing);
        console.log('server metric request_id:', timing.requestId);
        console.groupEnd();
      });
    });
  }, [reportStatus, reportHtml]);

  // 아코디언 토글 함수
  const toggleAccordion = (id: number) => {
    setExpandedProfileId(prevId => (prevId === id ? null : id));
  };

  // 진동 수동 조작 기능 (서버 API 연동)
  const handleVibration = async (workerId: string, direction: string) => {
    try {
      const result = await sendAlert(workerId, direction);
      if (result.status === 'success') {
        alert(`[작업자 ${workerId}] '${direction}' 진동 신호 전송 성공`);
      } else {
        alert(`진동 신호 전송 실패: ${JSON.stringify(result)}`);
      }
    } catch (err) {
      alert('서버 연결 실패. 서버가 실행 중인지 확인하세요.');
    }
  };

  // PDF 출력 기능 연동 (멀티페이지 지원)
  const handleDownloadPdf = async () => {
    const target = reportRef.current || document.getElementById('pdf-zone');
    if (!target) {
      console.error("PDF 변환 에러: 타겟 DOM 요소를 찾을 수 없습니다.");
      return;
    }

    try {
      const filter = (node: HTMLElement) => {
        return node.getAttribute ? node.getAttribute('data-html2canvas-ignore') !== 'true' : true;
      };

      const el = target as HTMLElement;
      const pixelRatio = 2;

      // overflow-y-auto로 잘린 영역까지 포함해 전체 스크롤 내용을 캡처하기 위해
      // 캡처 직전에 컨테이너 높이/overflow 제약을 일시적으로 해제한다.
      const originalStyles = {
        maxHeight: el.style.maxHeight,
        height: el.style.height,
        overflow: el.style.overflow,
        overflowY: el.style.overflowY,
      };
      el.style.maxHeight = 'none';
      el.style.height = 'auto';
      el.style.overflow = 'visible';
      el.style.overflowY = 'visible';

      const fullWidth = el.scrollWidth;
      const fullHeight = el.scrollHeight;

      let imgData: string;
      try {
        imgData = await toPng(el, {
          cacheBust: true,
          pixelRatio,
          filter,
          backgroundColor: '#ffffff',
          width: fullWidth,
          height: fullHeight,
          style: { padding: '24px' },
        });
      } finally {
        el.style.maxHeight = originalStyles.maxHeight;
        el.style.height = originalStyles.height;
        el.style.overflow = originalStyles.overflow;
        el.style.overflowY = originalStyles.overflowY;
      }

      const pdf = new jsPDF('p', 'mm', 'a4');
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 8;
      const usableWidth = pageWidth - margin * 2;
      const usableHeight = pageHeight - margin * 2;

      // 이미지 실제 픽셀 크기 구하기
      const img = new Image();
      img.src = imgData;
      await new Promise<void>((resolve) => { img.onload = () => resolve(); });

      const imgWidthPx = img.naturalWidth;
      const imgHeightPx = img.naturalHeight;

      // mm 단위로 환산된 전체 이미지 높이
      const totalImgHeightMm = (imgHeightPx * usableWidth) / imgWidthPx;

      if (totalImgHeightMm <= usableHeight) {
        // 한 페이지에 들어가면 그냥 넣기
        pdf.addImage(imgData, 'PNG', margin, margin, usableWidth, totalImgHeightMm);
      } else {
        // 멀티페이지: canvas 슬라이싱
        const canvas = document.createElement('canvas');
        canvas.width = imgWidthPx;
        canvas.height = imgHeightPx;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0);

        // 한 페이지에 들어갈 이미지 픽셀 높이
        const sliceHeightPx = Math.floor((usableHeight * imgWidthPx) / usableWidth);
        let yOffset = 0;
        let page = 0;

        while (yOffset < imgHeightPx) {
          const remaining = imgHeightPx - yOffset;
          const currentSlice = Math.min(sliceHeightPx, remaining);

          const sliceCanvas = document.createElement('canvas');
          sliceCanvas.width = imgWidthPx;
          sliceCanvas.height = currentSlice;
          const sliceCtx = sliceCanvas.getContext('2d')!;
          sliceCtx.fillStyle = '#ffffff';
          sliceCtx.fillRect(0, 0, imgWidthPx, currentSlice);
          sliceCtx.drawImage(canvas, 0, yOffset, imgWidthPx, currentSlice, 0, 0, imgWidthPx, currentSlice);

          const sliceData = sliceCanvas.toDataURL('image/png');
          const sliceHeightMm = (currentSlice * usableWidth) / imgWidthPx;

          if (page > 0) pdf.addPage();
          pdf.addImage(sliceData, 'PNG', margin, margin, usableWidth, sliceHeightMm);

          yOffset += currentSlice;
          page++;
        }
      }

      pdf.save(`${selectedDate}_SafeAI_안전사고_기록부.pdf`);
    } catch (error) {
      console.error("PDF 변환 에러 상세:", error);
      alert('PDF 변환 중 문제가 발생했습니다. 브라우저 개발자 도구를 확인해 주세요.');
    }
  };

  return (
    <div className="flex h-screen bg-[#f8fafc] font-sans text-slate-800 overflow-hidden">

      {/* =========================================================
          1. Left Sidebar
          ========================================================= */}
      <aside className="w-64 bg-white border-r border-slate-200 flex flex-col shrink-0 shadow-[4px_0_24px_rgba(0,0,0,0.02)] z-20 relative">
        <div className="p-8 pb-4">
          <div className="flex items-center gap-3 mb-12">
            <div className="bg-blue-600 p-2.5 rounded-2xl text-white shadow-lg shadow-blue-500/30">
              <ShieldCheck size={24} className="stroke-[2.5]" />
            </div>
            <div>
              <h1 className="font-extrabold text-xl text-slate-900 tracking-tight">SafeAI</h1>
              <p className="text-xs uppercase text-blue-600 font-bold tracking-widest mt-0.5">Control Tower</p>
            </div>
          </div>

          <nav className="space-y-3">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`w-full flex items-center gap-3 px-5 py-4 rounded-2xl text-[15px] font-bold transition-all duration-200 ${activeTab === 'dashboard'
                  ? 'bg-blue-50 text-blue-700 shadow-sm border border-blue-100 ring-2 ring-blue-500/10'
                  : 'text-slate-500 hover:bg-slate-50 hover:text-slate-900 border border-transparent'
                }`}
            >
              <LayoutDashboard size={20} /> 통합 대쉬보드
            </button>
            <button
              onClick={() => setActiveTab('records')}
              className={`w-full flex items-center gap-3 px-5 py-4 rounded-2xl text-[15px] font-bold transition-all duration-200 ${activeTab === 'records'
                  ? 'bg-blue-50 text-blue-700 shadow-sm border border-blue-100 ring-2 ring-blue-500/10'
                  : 'text-slate-500 hover:bg-slate-50 hover:text-slate-900 border border-transparent'
                }`}
            >
              <ClipboardList size={20} /> 일자별 기록
            </button>
            <button
              onClick={() => setActiveTab('vibration')}
              className={`w-full flex items-center gap-3 px-5 py-4 rounded-2xl text-[15px] font-bold transition-all duration-200 ${activeTab === 'vibration'
                  ? 'bg-purple-50 text-purple-700 shadow-sm border border-purple-100 ring-2 ring-purple-500/10'
                  : 'text-slate-500 hover:bg-slate-50 hover:text-slate-900 border border-transparent'
                }`}
            >
              <Radio size={20} /> 진동 수동 조작
            </button>
          </nav>
        </div>
      </aside>

      {/* =========================================================
          Main Area
          ========================================================= */}
      <div className="flex-1 flex flex-col h-full overflow-hidden relative">

        {/* === 2. Top Header Panel === */}
        <header className="flex justify-between items-center px-10 h-24 bg-white border-b border-slate-200 shrink-0 shadow-sm z-10 w-full">
          {/* 좌측에 텍스트와 달력 UI를 묶어 gap-4로 렌더링 */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <Calendar size={28} className="text-blue-500" />
              <h2 className="text-2xl font-black text-slate-900 tracking-tight">
                데이터 조회 기준일 : <span className="text-blue-600 ml-2">{selectedDate}</span>
              </h2>
            </div>

            {/* 시각적으로 투명한 Input을 커스텀 아이콘 박스 위에 덮어씌워서 텍스트 숨김 */}
            <div className="relative w-12 h-12 bg-slate-50 border border-slate-200 rounded-xl flex items-center justify-center hover:bg-white hover:border-blue-400 hover:shadow-md transition-all shadow-sm cursor-pointer group">
              <Calendar size={20} className="text-slate-500 group-hover:text-blue-600 transition-colors" />
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer [&::-webkit-calendar-picker-indicator]:absolute [&::-webkit-calendar-picker-indicator]:inset-0 [&::-webkit-calendar-picker-indicator]:w-full [&::-webkit-calendar-picker-indicator]:h-full [&::-webkit-calendar-picker-indicator]:cursor-pointer z-10"
                title="날짜 선택하기"
              />
            </div>
          </div>
        </header>

        {/* === 3. Scrollable Content Body === */}
        <main className="flex-1 overflow-y-auto p-10 bg-[#f8fafc]">

          {/* =====================================================
              페이지 1: 통합 대쉬보드 뷰
              ===================================================== */}
          {activeTab === 'dashboard' && (
            <div className="max-w-[1400px] mx-auto space-y-8 animate-in fade-in duration-500">

              {/* 상단 3개 요약 스탯 카드 (선택 날짜 기준 DB 집계) */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white rounded-[1.5rem] p-6 border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
                  <div className="flex justify-between items-start mb-4">
                    <p className="text-[15px] font-bold text-slate-500">선택 날짜 위험 로그</p>
                    <div className="bg-rose-50 p-3 rounded-xl text-rose-500"><AlertTriangle size={20} /></div>
                  </div>
                  <h3 className="text-4xl font-black text-slate-800">{totalRiskCount}<span className="text-xl font-bold text-slate-400 ml-1.5">건</span></h3>
                  <p className={`mt-3 text-xs font-bold ${summaryStatus === 'error' ? 'text-red-400' : 'text-slate-400'}`}>
                    {summaryStatus === 'loading' ? 'DB 집계 동기화 중' : summaryStatus === 'error' ? 'DB 집계 조회 실패' : `${selectedDate} 기준`}
                  </p>
                </div>
                <div className="bg-white rounded-[1.5rem] p-6 border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
                  <div className="flex justify-between items-start mb-4">
                    <p className="text-[15px] font-bold text-slate-500">Warning</p>
                    <div className="bg-amber-50 p-3 rounded-xl text-amber-600"><Zap size={20} /></div>
                  </div>
                  <h3 className="text-4xl font-black text-slate-800">{totalWarningCount}<span className="text-xl font-bold text-slate-400 ml-1.5">건</span></h3>
                  <p className="mt-3 text-xs font-bold text-slate-400">주의 단계 위험 예측</p>
                </div>
                <div className="bg-white rounded-[1.5rem] p-6 border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
                  <div className="flex justify-between items-start mb-4">
                    <p className="text-[15px] font-bold text-slate-500">Danger</p>
                    <div className="bg-red-50 p-3 rounded-xl text-red-600"><Eye size={20} /></div>
                  </div>
                  <h3 className="text-4xl font-black text-slate-800">{totalDangerCount}<span className="text-xl font-bold text-slate-400 ml-1.5">건</span></h3>
                  <p className="mt-3 text-xs font-bold text-slate-400">긴급 단계 위험 예측</p>
                </div>
              </div>

              {/* 시간대별 추세 및 작업자 스코어 패널 */}
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 items-start">

                {/* 1. 시간대별 위험 추이 그래프 (Recharts) */}
                <div className="bg-white rounded-[1.5rem] p-8 border border-slate-200 shadow-sm flex flex-col h-[520px]">
                  <div className="flex items-center justify-between mb-8">
                    <h3 className="text-xl font-bold text-slate-900 flex items-center gap-2">
                      <Activity size={22} className="text-blue-500 stroke-[2.5]" />
                      시간대별 위험 발생 추이
                    </h3>
                  </div>
                  <div className="flex-1 w-full relative">
                    <ResponsiveContainer width="100%" height="85%">
                      {/* 버그 수정: bottom 마진을 25로 늘려 X축 텍스트 잘림 현상 방지 */}
                      <LineChart data={lineChartData} margin={{ top: 10, right: 30, left: -20, bottom: 25 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                        <XAxis
                          dataKey="time"
                          axisLine={false}
                          tickLine={false}
                          tick={{ fill: '#64748b', fontSize: 13, fontWeight: 700 }}
                          dy={15}
                        />
                        <YAxis
                          axisLine={false}
                          tickLine={false}
                          tick={{ fill: '#64748b', fontSize: 13, fontWeight: 700 }}
                          dx={-10}
                        />
                        <Tooltip
                          contentStyle={{ borderRadius: '1rem', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)', fontWeight: 'bold' }}
                        />
                        <Line
                          type="monotone"
                          dataKey="count"
                          stroke="#3b82f6"
                          strokeWidth={4}
                          dot={{ r: 6, fill: '#3b82f6', strokeWidth: 3, stroke: '#fff' }}
                          activeDot={{ r: 9, strokeWidth: 0 }}
                          animationDuration={2000}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                    <div className="mt-8 pt-4 border-t border-slate-100 flex justify-end">
                      <span className="text-xs font-bold text-slate-400 uppercase tracking-widest bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200">X : 시간 / Y : 위험 건수(건)</span>
                    </div>
                  </div>
                </div>

                {/* 2. 작업자 안전 스코어 (아코디언 클릭 인터랙션) */}
                <div className="bg-white rounded-[1.5rem] p-8 border border-slate-200 shadow-sm flex flex-col">
                  <div className="mb-6">
                    <h3 className="text-xl font-bold text-slate-900 flex items-center gap-2">
                      <User size={22} className="text-blue-500 stroke-[2.5]" />
                      작업자 안전도 평가 명단
                    </h3>
                  </div>

                  <div className="flex flex-col space-y-3">
                    {profiles.map(profile => {
                      const isSafe = profile.score >= 75;
                      const isExpanded = expandedProfileId === profile.id;

                      const scoreBoxBg = isSafe ? 'bg-emerald-50 text-emerald-600 border border-emerald-100' : 'bg-red-50 text-red-600 border border-red-100';
                      const interactiveBg = isSafe ? 'hover:bg-emerald-50/50' : 'hover:bg-red-50/50';

                      return (
                        <div key={profile.id} className="flex flex-col border border-slate-100 rounded-[1.25rem] overflow-hidden transition-all shadow-sm">

                          <div
                            onClick={() => toggleAccordion(profile.id)}
                            className={`flex items-center justify-between p-4 cursor-pointer transition-colors duration-200 ${interactiveBg} ${isExpanded ? 'bg-slate-50 border-b border-slate-100' : 'bg-white'}`}
                          >
                            <div className="flex items-center gap-4">
                              <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center shadow-sm border border-slate-200 overflow-hidden">
                                <img src={`https://api.dicebear.com/7.x/notionists/svg?seed=${profile.name}`} alt="avatar" className="w-full h-full" />
                              </div>
                              <h4 className="text-[17px] font-black text-slate-800">{profile.name}</h4>
                            </div>

                            <div className="flex items-center gap-4">
                              <div className={`px-5 py-2.5 rounded-xl flex items-center gap-2 font-black text-xl shadow-sm ${scoreBoxBg}`}>
                                {profile.count}
                                <span className="text-sm font-bold opacity-60">건</span>
                              </div>
                              <ChevronDown size={20} className={`text-slate-400 transition-transform duration-300 ${isExpanded ? 'rotate-180' : 'rotate-0'}`} />
                            </div>
                          </div>

                          <div
                            className={`transition-all duration-300 ease-in-out overflow-hidden ${isExpanded ? 'max-h-60 opacity-100' : 'max-h-0 opacity-0'
                              }`}
                          >
                            <div className="p-5 bg-slate-50">
                              <p className="text-sm font-bold text-slate-500 mb-3 px-1">위험 알림 상세</p>
                              {profile.violations.length > 0 ? (
                                <ul className="space-y-2">
                                  {profile.violations.map((v, i) => (
                                    <li key={i} className="flex items-center gap-3 text-[14px] font-bold text-slate-700 bg-white border border-slate-200 px-4 py-3 rounded-xl shadow-sm">
                                      <AlertTriangle size={16} className="text-amber-500" />
                                      {v}
                                    </li>
                                  ))}
                                </ul>
                              ) : (
                                <div className="flex items-center gap-2 text-[14px] font-bold text-emerald-600 bg-emerald-50 border border-emerald-100 px-4 py-3 rounded-xl">
                                  <ShieldCheck size={18} /> {selectedDate} 위험 알림 없음 · 전체 누적 {profile.allTimeCount}회
                                </div>
                              )}
                            </div>
                          </div>

                        </div>
                      );
                    })}
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* =====================================================
              페이지 2: 일자별 기록 뷰 
              ===================================================== */}
          {activeTab === 'records' && (
            <div className="max-w-[1400px] mx-auto h-[calc(100vh-160px)] flex flex-col animate-in fade-in duration-500">

              <div
                className="flex-1 bg-white rounded-[2rem] p-10 border border-slate-200 shadow-sm relative flex flex-col min-h-0"
              >
                <div className="mb-6 pb-6 border-b border-slate-100 flex justify-between items-center shrink-0">
                  <div>
                    <h3 className="text-2xl font-black text-slate-900 flex items-center gap-3">
                      <ScrollText size={28} className="text-blue-600" />
                      일일 이벤트 증적 기록부
                    </h3>
                    <p className="text-[15px] text-slate-500 mt-2 font-semibold">데이터베이스에서 동기화된 리포트 원본 HTML 컨테이너입니다.</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="px-4 py-2 bg-slate-50 border border-slate-200 rounded-xl text-sm font-bold text-slate-600 h-fit">
                      {selectedDate} ARCHIVE
                    </div>
                    <button
                      onClick={handleDownloadPdf}
                      disabled={reportStatus !== 'ok'}
                      data-html2canvas-ignore="true"
                      className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-bold transition-all text-sm h-fit ${
                        reportStatus === 'ok'
                          ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-md hover:shadow-blue-500/30 ring-2 ring-blue-500/20'
                          : 'bg-slate-200 text-slate-400 cursor-not-allowed'
                      }`}
                    >
                      <Download size={18} />
                      PDF 다운로드
                    </button>
                  </div>
                </div>

                {/* 콘텐츠 영역: 상태에 따라 분기 */}
                {reportStatus === 'ok' ? (
                  <>
                    {USE_MOCK_DATA ? (
                      <div className="flex-1 w-full bg-slate-50/50 rounded-2xl border-2 border-dashed border-slate-300 p-8 overflow-y-auto print-container min-h-0 mt-4 h-full">
                        <div ref={reportRef} id="pdf-zone" className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
                          <div className="p-6 border-b border-slate-100 mb-2">
                            <h2 className="text-xl font-extrabold text-slate-800">일일 안전 사고 증적 리포트 - {selectedDate}</h2>
                            <p className="text-sm font-bold text-slate-500 mt-1">현장 스마트 안전 관리 시스템 자동 생성 요약</p>
                          </div>
                          <div className="p-6 space-y-4">
                            {MOCK_INCIDENTS.map((inc, i) => (
                              <div key={i} className="flex flex-col border border-slate-200 rounded-xl p-5 bg-slate-50 shadow-sm relative">
                                <div className="flex justify-between items-center mb-4">
                                  <div className="flex items-center gap-2">
                                    <AlertTriangle size={18} className="text-slate-400" />
                                    <span className="font-bold text-slate-500 text-sm">{inc.time}</span>
                                  </div>
                                  <span className={`px-3 py-1 text-xs font-black rounded-lg border ${inc.severityColor}`}>
                                    {inc.severity}
                                  </span>
                                </div>
                                <div className="space-y-2">
                                  <div className="flex text-[15px]">
                                    <span className="font-black text-slate-600 w-28 shrink-0">관련 작업자 :</span>
                                    <span className="font-bold text-slate-800">{inc.worker}</span>
                                  </div>
                                  <div className="flex text-[15px]">
                                    <span className="font-black text-slate-600 w-28 shrink-0">이벤트 내용 :</span>
                                    <span className="font-bold text-red-500">{inc.event}</span>
                                  </div>
                                  <div className="flex text-[15px]">
                                    <span className="font-black text-slate-600 w-28 shrink-0">조치 결과 :</span>
                                    <span className="font-bold text-blue-600">{inc.action}</span>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div
                        ref={reportRef}
                        id="pdf-zone"
                        className="flex-1 w-full bg-slate-50/50 rounded-2xl border-2 border-dashed border-slate-300 p-8 overflow-y-auto print-container min-h-0 mt-4 font-semibold text-slate-500 leading-relaxed"
                        dangerouslySetInnerHTML={{ __html: reportHtml }}
                      ></div>
                    )}
                  </>
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center mt-4 rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50/30">

                    {reportStatus === 'loading' ? (
                      <>
                        <div className="flex items-center gap-2 mb-4">
                          {[0, 1, 2].map((i) => (
                            <div
                              key={i}
                              className="w-2.5 h-2.5 rounded-full bg-slate-300"
                              style={{
                                animation: 'toss-bounce 1.4s infinite ease-in-out both',
                                animationDelay: `${i * 0.16}s`,
                              }}
                            />
                          ))}
                        </div>
                        <p className="text-lg font-black text-slate-400 tracking-tight">리포트를 불러오는 중...</p>
                      </>

                    ) : reportStatus === 'error' ? (
                      <>
                        <div className="w-20 h-20 mb-6 rounded-3xl bg-red-50 border border-red-100 flex items-center justify-center">
                          <AlertTriangle size={32} className="text-red-400" />
                        </div>
                        <p className="text-xl font-black text-slate-700 tracking-tight">서버에 연결할 수 없어요</p>
                        <p className="text-[15px] font-semibold text-slate-400 mt-3">백엔드 서버가 실행 중인지 확인해주세요</p>
                        <div className="mt-5 px-5 py-2.5 bg-red-50 rounded-full border border-red-100">
                          <p className="text-xs font-bold text-red-400">네트워크 또는 서버 상태를 점검해주세요</p>
                        </div>
                      </>

                    ) : reportStatus === 'generating' ? (
                      <>
                        {/* 토스 스타일 — 펄스 아이콘 + 바운싱 도트 */}
                        <style>{`
                          @keyframes toss-bounce {
                            0%, 80%, 100% { transform: scale(0); opacity: 0.4; }
                            40% { transform: scale(1); opacity: 1; }
                          }
                          @keyframes toss-pulse-ring {
                            0% { transform: scale(0.8); opacity: 0.6; }
                            50% { transform: scale(1.15); opacity: 0.2; }
                            100% { transform: scale(0.8); opacity: 0.6; }
                          }
                        `}</style>
                        <div className="relative w-24 h-24 mb-8 flex items-center justify-center">
                          <div className="absolute inset-0 rounded-full bg-blue-100" style={{ animation: 'toss-pulse-ring 2s infinite ease-in-out' }} />
                          <div className="absolute inset-2 rounded-full bg-blue-50" style={{ animation: 'toss-pulse-ring 2s infinite ease-in-out 0.3s' }} />
                          <div className="relative w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/30">
                            <Sparkles size={26} className="text-white" />
                          </div>
                        </div>
                        <p className="text-xl font-black text-slate-700 tracking-tight">리포트를 생성하고 있어요</p>
                        <p className="text-[15px] font-semibold text-slate-400 mt-3">
                          <span className="text-blue-500 font-bold">{selectedDate}</span> 데이터를 AI가 분석하고 있습니다
                        </p>
                        <div className="flex items-center gap-2 mt-6">
                          {[0, 1, 2].map((i) => (
                            <div
                              key={i}
                              className="w-2 h-2 rounded-full bg-blue-400"
                              style={{
                                animation: 'toss-bounce 1.4s infinite ease-in-out both',
                                animationDelay: `${i * 0.16}s`,
                              }}
                            />
                          ))}
                        </div>
                        <div className="mt-5 px-5 py-2.5 bg-slate-100 rounded-full">
                          <p className="text-xs font-bold text-slate-400">잠시만 기다려주세요</p>
                        </div>
                      </>

                    ) : reportStatus === 'generate_no_data' ? (
                      <>
                        <div className="w-20 h-20 mb-6 rounded-3xl bg-amber-50 border border-amber-100 flex items-center justify-center">
                          <FileWarning size={32} className="text-amber-400" />
                        </div>
                        <p className="text-xl font-black text-slate-700 tracking-tight">데이터가 없어 생성할 수 없어요</p>
                        <p className="text-[15px] font-semibold text-slate-400 mt-3">
                          <span className="text-blue-500 font-bold">{selectedDate}</span>에 수집된 사고 로그가 없습니다
                        </p>
                        <div className="mt-5 px-5 py-2.5 bg-amber-50 rounded-full border border-amber-100">
                          <p className="text-xs font-bold text-amber-500">다른 날짜를 선택해주세요</p>
                        </div>
                      </>

                    ) : (
                      /* empty — 기본 페이지: 리포트 생성 유도 */
                      <>
                        <div className="w-20 h-20 mb-6 rounded-3xl bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-100 flex items-center justify-center">
                          <NotebookPen size={32} className="text-blue-500" />
                        </div>
                        <p className="text-xl font-black text-slate-700 tracking-tight">AI 리포트를 생성해보세요</p>
                        <p className="text-[15px] font-semibold text-slate-400 mt-3">
                          <span className="text-blue-500 font-bold">{selectedDate}</span>의 사고 기록을 AI가 자동으로 요약합니다
                        </p>
                        <button
                          onClick={handleGenerateReport}
                          className="mt-7 flex items-center gap-2.5 px-7 py-3.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-bold text-[15px] rounded-2xl shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 transition-all duration-200 active:scale-95"
                        >
                          <Sparkles size={18} />
                          리포트 생성하기
                        </button>
                        <p className="text-xs font-semibold text-slate-300 mt-4">로컬 LLM을 사용하여 일일 리포트를 자동 생성합니다</p>
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* =====================================================
              페이지 3: 진동 수동 조작 뷰
              ===================================================== */}
          {activeTab === 'vibration' && (
            <div className="max-w-[1400px] mx-auto flex flex-col animate-in fade-in duration-500 mb-10">
              <div className="bg-white rounded-[2rem] p-10 border border-slate-200 shadow-sm flex flex-col">
                <div className="mb-8 pb-6 border-b border-slate-100 flex justify-between items-center shrink-0">
                  <div>
                    <h3 className="text-2xl font-black text-slate-900 flex items-center gap-3">
                      <Radio size={28} className="text-purple-600" />
                      작업자 진동 수동 조작
                    </h3>
                    <p className="text-[15px] text-slate-500 mt-2 font-semibold">
                      각 작업자별 스마트 안전조끼 진동 모터를 수동으로 작동시켜 신호를 전달할 수 있습니다.
                    </p>
                  </div>
                </div>

                <div className="flex flex-col space-y-4">
                  {profiles.map((profile) => (
                    <div key={profile.id} className="flex flex-col md:flex-row md:items-center justify-between p-6 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-2xl transition-colors duration-200 group">
                      <div className="flex items-center gap-5 mb-4 md:mb-0">
                        <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center shadow-sm border border-slate-200 overflow-hidden shrink-0 group-hover:scale-110 transition-transform">
                          <Radio size={20} className="text-slate-500 group-hover:text-purple-600 transition-colors" />
                        </div>
                        <div>
                          <h4 className="text-lg font-black text-slate-800 tracking-tight">작업자 {profile.id}</h4>
                          <p className="text-[14px] font-bold text-slate-500 mt-0.5">
                            상태: <span className="text-emerald-500">정상 (Connected)</span>
                          </p>
                        </div>
                      </div>

                      <div className="flex items-center gap-3">
                        {[
                          { label: 'LEFT', value: 'left' },
                          { label: 'RIGHT', value: 'right' },
                          { label: 'BACK', value: 'back' },
                          { label: 'ALL', value: 'all' },
                        ].map((dir) => (
                          <button
                            key={dir.value}
                            onClick={() => handleVibration(String(profile.id), dir.value)}
                            className="px-5 py-3 min-w-[80px] rounded-xl text-[15px] font-bold text-slate-700 bg-white border border-slate-300 shadow-sm hover:text-white hover:bg-purple-600 hover:border-purple-600 hover:shadow-lg hover:shadow-purple-500/30 active:scale-95 active:bg-purple-700 transition-all duration-200"
                          >
                            {dir.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

        </main>
      </div>
    </div>
  );
}
