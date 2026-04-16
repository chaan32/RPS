import React, { useState, useRef, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import {
  LayoutDashboard,
  FileText,
  Calendar,
  Download,
  AlertTriangle,
  Activity,
  User,
  ShieldCheck,
  Zap,
  Radio,
  Eye,
  ChevronDown
} from 'lucide-react';
import html2canvas from 'html2canvas';
import { jsPDF } from 'jspdf';
import { fetchReports, sendAlert } from '../api';
import type { Report } from '../api';

// === Mock Data ===
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

const initialProfiles = [
  { id: 1, name: "작업자 1", score: 85, violations: ["크레인 드롭존 침범 1회", "지게차 접근 경고 1회"] },
  { id: 2, name: "작업자 2", score: 95, violations: ["지게차 사각지대 진입 1회"] },
  { id: 3, name: "작업자 3", score: 70, violations: ["크레인 파단음 구역 이탈 지연 1회", "사각지대 진입 2회"] },
  { id: 4, name: "작업자 4", score: 60, violations: ["크레인 작업 반경 내 보행 2회", "교차로 지게차 충돌 경보 1회"] },
  { id: 5, name: "작업자 5", score: 100, violations: [] }
];

export default function DailyAdminDashboard() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'records' | 'vibration'>('dashboard');

  // === 전역(Global) 상태 및 헤더(Top Panel) 연동 ===
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().slice(0, 10));

  // 아코디언 열림/닫힘 관리를 위한 상태
  const [expandedProfileId, setExpandedProfileId] = useState<number | null>(null);

  const [profiles, setProfiles] = useState(initialProfiles);

  // === 일자별 기록 뷰: 서버 리포트 연동 ===
  const [reports, setReports] = useState<Report[]>([]);
  const [reportHtml, setReportHtml] = useState('');
  const [reportStatus, setReportStatus] = useState<'loading' | 'empty' | 'error' | 'ok'>('loading');

  useEffect(() => {
    fetchReports()
      .then((data) => {
        setReports(data);
        const matched = data.find((r) => r.date === selectedDate);
        if (matched) {
          setReportHtml(matched.contents);
          setReportStatus('ok');
        } else {
          setReportStatus('empty');
        }
      })
      .catch(() => {
        setReportStatus('error');
      });
  }, []);

  // 날짜 변경 시 해당 날짜 리포트로 전환
  useEffect(() => {
    const matched = reports.find((r) => r.date === selectedDate);
    if (matched) {
      setReportHtml(matched.contents);
      setReportStatus('ok');
    } else if (reports.length > 0 || reportStatus !== 'loading') {
      setReportStatus('empty');
    }
  }, [selectedDate, reports]);

  const reportRef = useRef<HTMLDivElement>(null);

  // 아코디언 토글 함수
  const toggleAccordion = (id: number) => {
    setExpandedProfileId(prevId => (prevId === id ? null : id));
  };

  // 진동 수동 조작 기능 (서버 API 연동)
  const handleVibration = async (makerId: string, direction: string) => {
    try {
      const result = await sendAlert(makerId, direction);
      if (result.status === 'success') {
        alert(`[작업자 ${makerId}] '${direction}' 진동 신호 전송 성공`);
      } else {
        alert(`진동 신호 전송 실패: ${JSON.stringify(result)}`);
      }
    } catch (err) {
      alert('서버 연결 실패. 서버가 실행 중인지 확인하세요.');
    }
  };

  // PDF 출력 기능 연동
  const handleDownloadPdf = async () => {
    if (!reportRef.current) return;
    try {
      const canvas = await html2canvas(reportRef.current, { scale: 2, useCORS: true });
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save(`safety_report_${selectedDate}.pdf`);
    } catch (error) {
      console.error('PDF 다운로드 에러:', error);
      alert('PDF 변환 중 문제가 발생했습니다.');
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
              <FileText size={20} /> 일자별 기록
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

              {/* 상단 3개 요약 스탯 카드 (4번째 카드 통신상태 삭제 후 grid-cols-3) */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white rounded-[1.5rem] p-6 border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
                  <div className="flex justify-between items-start mb-4">
                    <p className="text-[15px] font-bold text-slate-500">일일 총 발생 위험</p>
                    <div className="bg-rose-50 p-3 rounded-xl text-rose-500"><AlertTriangle size={20} /></div>
                  </div>
                  <h3 className="text-4xl font-black text-slate-800">3<span className="text-xl font-bold text-slate-400 ml-1.5">건</span></h3>
                </div>
                <div className="bg-white rounded-[1.5rem] p-6 border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
                  <div className="flex justify-between items-start mb-4">
                    <p className="text-[15px] font-bold text-slate-500">크레인 이상</p>
                    <div className="bg-purple-50 p-3 rounded-xl text-purple-600"><Zap size={20} /></div>
                  </div>
                  <h3 className="text-4xl font-black text-slate-800">2<span className="text-xl font-bold text-slate-400 ml-1.5">건</span></h3>
                </div>
                <div className="bg-white rounded-[1.5rem] p-6 border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
                  <div className="flex justify-between items-start mb-4">
                    <p className="text-[15px] font-bold text-slate-500">지게차 이상</p>
                    <div className="bg-indigo-50 p-3 rounded-xl text-indigo-600"><Eye size={20} /></div>
                  </div>
                  <h3 className="text-4xl font-black text-slate-800">1<span className="text-xl font-bold text-slate-400 ml-1.5">건</span></h3>
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
                                {profile.score}
                                <span className="text-sm font-bold opacity-60">점</span>
                              </div>
                              <ChevronDown size={20} className={`text-slate-400 transition-transform duration-300 ${isExpanded ? 'rotate-180' : 'rotate-0'}`} />
                            </div>
                          </div>

                          <div
                            className={`transition-all duration-300 ease-in-out overflow-hidden ${isExpanded ? 'max-h-60 opacity-100' : 'max-h-0 opacity-0'
                              }`}
                          >
                            <div className="p-5 bg-slate-50">
                              <p className="text-sm font-bold text-slate-500 mb-3 px-1">위반 내역 상세</p>
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
                                  <ShieldCheck size={18} /> 위반 내역이 없습니다 (안전 우수자)
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
                ref={reportRef}
                className="flex-1 bg-white rounded-[2rem] p-10 border border-slate-200 shadow-sm relative flex flex-col min-h-0"
              >
                <div className="mb-6 pb-6 border-b border-slate-100 flex justify-between items-center shrink-0">
                  <div>
                    <h3 className="text-2xl font-black text-slate-900 flex items-center gap-3">
                      <FileText size={28} className="text-blue-600" />
                      일일 이벤트 증적 기록부
                    </h3>
                    <p className="text-[15px] text-slate-500 mt-2 font-semibold">데이터베이스에서 동기화된 리포트 원본 HTML 컨테이너입니다.</p>
                  </div>
                  <div className="px-4 py-2 bg-slate-50 border border-slate-200 rounded-xl text-sm font-bold text-slate-600 h-fit">
                    {selectedDate} ARCHIVE
                  </div>
                </div>

                {/* 콘텐츠 영역: 상태에 따라 분기 */}
                {reportStatus === 'ok' ? (
                  <>
                    <div
                      className="flex-1 w-full bg-slate-50/50 rounded-2xl border-2 border-dashed border-slate-300 p-8 overflow-y-auto print-container min-h-0 mt-4 font-semibold text-slate-500 leading-relaxed"
                      dangerouslySetInnerHTML={{ __html: reportHtml }}
                    ></div>
                    <div className="absolute bottom-10 right-10 z-10">
                      <button
                        onClick={handleDownloadPdf}
                        className="flex items-center gap-3 bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-[1.25rem] shadow-xl hover:shadow-blue-500/40 transition-all font-black tracking-wide text-[15px] group ring-4 ring-blue-500/10"
                      >
                        <Download size={22} className="group-hover:-translate-y-1 transition-transform" />
                        PDF 리포트 저장
                      </button>
                    </div>
                  </>
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center mt-4 rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50/30">
                    {/* 일러스트 SVG */}
                    <svg width="180" height="160" viewBox="0 0 180 160" fill="none" className="mb-6 opacity-80">
                      {/* 빈 문서 */}
                      <rect x="50" y="20" width="80" height="105" rx="8" fill="#e2e8f0" stroke="#cbd5e1" strokeWidth="2"/>
                      <rect x="50" y="20" width="80" height="105" rx="8" fill="url(#docGrad)" stroke="#cbd5e1" strokeWidth="2"/>
                      {/* 접힌 모서리 */}
                      <path d="M110 20 L130 20 L130 40 Z" fill="#f1f5f9" stroke="#cbd5e1" strokeWidth="1.5" strokeLinejoin="round"/>
                      <path d="M110 20 L110 40 L130 40" fill="#e2e8f0" stroke="#cbd5e1" strokeWidth="1.5" strokeLinejoin="round"/>
                      {/* 텍스트 라인 placeholder */}
                      <rect x="66" y="52" width="48" height="5" rx="2.5" fill="#cbd5e1" opacity="0.7"/>
                      <rect x="66" y="64" width="38" height="5" rx="2.5" fill="#cbd5e1" opacity="0.5"/>
                      <rect x="66" y="76" width="44" height="5" rx="2.5" fill="#cbd5e1" opacity="0.4"/>
                      <rect x="66" y="88" width="30" height="5" rx="2.5" fill="#cbd5e1" opacity="0.3"/>
                      {/* 돋보기 */}
                      <circle cx="128" cy="100" r="20" fill="none" stroke="#94a3b8" strokeWidth="3" opacity="0.6"/>
                      <line x1="142" y1="114" x2="156" y2="128" stroke="#94a3b8" strokeWidth="4" strokeLinecap="round" opacity="0.6"/>
                      {/* 물음표 */}
                      <text x="122" y="107" textAnchor="middle" fontSize="20" fontWeight="800" fill="#94a3b8" opacity="0.7">?</text>
                      <defs>
                        <linearGradient id="docGrad" x1="50" y1="20" x2="130" y2="125">
                          <stop offset="0%" stopColor="#f8fafc"/>
                          <stop offset="100%" stopColor="#e2e8f0"/>
                        </linearGradient>
                      </defs>
                    </svg>

                    {reportStatus === 'loading' ? (
                      <>
                        <p className="text-lg font-black text-slate-400 tracking-tight">리포트를 불러오는 중...</p>
                        <p className="text-sm font-semibold text-slate-300 mt-2">서버에서 데이터를 가져오고 있습니다</p>
                      </>
                    ) : reportStatus === 'error' ? (
                      <>
                        <p className="text-lg font-black text-red-400 tracking-tight">서버 연결 실패</p>
                        <p className="text-sm font-semibold text-slate-400 mt-2">백엔드 서버가 실행 중인지 확인해주세요</p>
                      </>
                    ) : (
                      <>
                        <p className="text-lg font-black text-slate-400 tracking-tight">해당 날짜의 리포트가 없습니다</p>
                        <p className="text-sm font-semibold text-slate-300 mt-2">
                          <span className="text-blue-400 font-bold">{selectedDate}</span> 에 생성된 기록이 존재하지 않습니다
                        </p>
                        <p className="text-xs font-semibold text-slate-300 mt-1">다른 날짜를 선택하거나, 리포트를 먼저 생성해주세요</p>
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
                  {[1, 2, 3, 4, 5].map((id) => (
                    <div key={id} className="flex flex-col md:flex-row md:items-center justify-between p-6 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-2xl transition-colors duration-200 group">
                      <div className="flex items-center gap-5 mb-4 md:mb-0">
                        <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center shadow-sm border border-slate-200 overflow-hidden shrink-0 group-hover:scale-110 transition-transform">
                          <Radio size={20} className="text-slate-500 group-hover:text-purple-600 transition-colors" />
                        </div>
                        <div>
                          <h4 className="text-lg font-black text-slate-800 tracking-tight">작업자 {id}</h4>
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
                            onClick={() => handleVibration(String(id), dir.value)}
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
