import React, { useState, useRef } from 'react';
import html2canvas from 'html2canvas';
import { jsPDF } from 'jspdf';
import { 
  Download, 
  Activity, 
  Users, 
  Clock,
  ShieldAlert,
  AlertTriangle,
  LayoutDashboard,
  FileText,
  Video,
  Waves,
  Eye,
  TrendingDown,
  TrendingUp,
  Camera
} from 'lucide-react';

// === [1] 모달 시나리오용 상태 타입 ===
type ModalData = {
  isOpen: boolean;
  event: any;
};

// === [2] 상세화된 멀티모달 기반 Mock Data ===
const mockEvents = [
  { 
    id: 1, 
    datetime: "14:30:22", 
    worker: "김현수", 
    equipment: "크레인", 
    type: "오디오", 
    analysis: "[파단음 경고] 105dB 비정상 마찰음 감지 (와이어 파손 의심)", 
    action: "작업 즉시 중단 및 안전 점검 지시" 
  },
  { 
    id: 2, 
    datetime: "15:10:05", 
    worker: "전성은", 
    equipment: "지게차", 
    type: "비전", 
    analysis: "[사각지대] 후방 보행자 충돌 위험 감지 (거리 1.5m)", 
    action: "경고음 발생 및 자동 감속 처리" 
  },
  { 
    id: 3, 
    datetime: "16:20:00", 
    worker: "시스템", 
    equipment: "크레인", 
    type: "비전", 
    analysis: "[드롭존 침범] 반경 3m 내 작업자 감지", 
    action: "현장 경고 알림 발송" 
  }
];

const mockProfiles = [
  { id: 1, name: "김현수", role: "크레인 신호수", score: 85, recentViolations: ["드롭존 침범 1회"], trend: [95, 90, 88, 85] },
  { id: 2, name: "전성은", role: "지게차 운전원", score: 95, recentViolations: ["사각지대 진입 1회"], trend: [80, 85, 90, 95] },
  { id: 3, name: "유희덕", role: "현장 소장", score: 100, recentViolations: [], trend: [100, 100, 100, 100] }
];

// 간단한 스파크라인(미니 차트) 컴포넌트
const Sparkline = ({ data }: { data: number[] }) => {
  return (
    <div className="flex items-end gap-1 h-8">
      {data.map((val, i) => (
        <div 
          key={i} 
          className={`w-2.5 rounded-t-sm transition-all duration-300 ${i === data.length - 1 ? 'bg-blue-500' : 'bg-slate-200'}`}
          style={{ height: `${val}%`, minHeight: '4px' }}
          title={`${val}점`}
        />
      ))}
    </div>
  );
};

export default function DailyAdminDashboard() {
  const [activeMenu, setActiveMenu] = useState('dashboard');
  const [modalData, setModalData] = useState<ModalData>({ isOpen: false, event: null });
  const dashboardRef = useRef<HTMLDivElement>(null);

  // PDF 출력 기능
  const handleDownloadPdf = async () => {
    if (!dashboardRef.current) return;
    try {
      const canvas = await html2canvas(dashboardRef.current, { scale: 2, useCORS: true });
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
      
      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save(`safety_control_tower_${new Date().toISOString().slice(0,10)}.pdf`);
    } catch (error) {
      console.error('PDF 다운로드 실패:', error);
      alert('PDF 생성 중 문제가 발생했습니다.');
    }
  };

  return (
    <div className="flex h-screen bg-slate-50 text-slate-800 font-sans overflow-hidden">
      
      {/* === 1. Left Sidebar === */}
      <aside className="w-64 bg-slate-100/80 border-r border-slate-200 flex flex-col hidden md:flex shrink-0">
        <div className="p-6 pb-2">
          <div className="flex items-center gap-2 mb-8">
            <div className="bg-blue-600 p-2 rounded-xl text-white shadow-sm">
              <ShieldAlert size={24} />
            </div>
            <div>
              <h1 className="font-bold text-lg leading-tight text-slate-900">SafeTower</h1>
              <p className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Multimodal AI</p>
            </div>
          </div>

          <nav className="space-y-1.5">
            <button onClick={() => setActiveMenu('dashboard')} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all ${activeMenu === 'dashboard' ? 'bg-white text-blue-700 shadow-sm border border-slate-200/60' : 'text-slate-500 hover:bg-slate-200/50 hover:text-slate-800'}`}>
              <LayoutDashboard size={18} /> 통합 대시보드
            </button>
            <button onClick={() => setActiveMenu('logs')} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all ${activeMenu === 'logs' ? 'bg-white text-blue-700 shadow-sm border border-slate-200/60' : 'text-slate-500 hover:bg-slate-200/50 hover:text-slate-800'}`}>
              <FileText size={18} /> 위험 상황 로그
            </button>
            <button onClick={() => setActiveMenu('workers')} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all ${activeMenu === 'workers' ? 'bg-white text-blue-700 shadow-sm border border-slate-200/60' : 'text-slate-500 hover:bg-slate-200/50 hover:text-slate-800'}`}>
              <Users size={18} /> 작업자 관리
            </button>
          </nav>
        </div>
      </aside>

      {/* === 2. Main Content Area === */}
      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Header */}
        <header className="flex justify-between items-center px-8 py-5 bg-white border-b border-slate-200 shrink-0 z-10">
          <div>
            <h2 className="text-xl font-bold text-slate-900">위험 관제 센터</h2>
            <p className="text-sm text-slate-500 font-medium">크레인 및 지게차 특화 모니터링 현황</p>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm font-bold text-slate-600 bg-slate-100 px-4 py-2 rounded-lg">
              {new Date().toLocaleDateString('ko-KR')}
            </span>
            <button 
              onClick={handleDownloadPdf}
              className="flex items-center gap-2 bg-slate-900 hover:bg-slate-800 text-white px-5 py-2.5 rounded-xl text-sm font-bold transition-all shadow-md hover:shadow-lg"
            >
              <Download size={16} /> PDF 리포트 출력
            </button>
          </div>
        </header>

        {/* Scrollable Content (PDF Capture Target) */}
        <div className="flex-1 overflow-y-auto p-8" ref={dashboardRef}>
          <div className="max-w-6xl mx-auto space-y-8">
            
            {/* 상단 4개 스탯 카드 */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex justify-between items-start mb-2">
                  <p className="text-sm font-bold text-slate-500">일일 총 발생 위험</p>
                  <div className="bg-red-50 p-2 rounded-lg text-red-500"><AlertTriangle size={18} /></div>
                </div>
                <div className="flex items-end gap-2">
                  <h3 className="text-3xl font-black text-slate-800">3<span className="text-lg font-bold text-slate-500 ml-1">건</span></h3>
                  <span className="flex items-center text-xs font-bold text-emerald-500 mb-1.5"><TrendingDown size={14} className="mr-0.5"/> -15%</span>
                </div>
              </div>
              <div className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex justify-between items-start mb-2">
                  <p className="text-sm font-bold text-slate-500">크레인 이상 (파단음/드롭존)</p>
                  <div className="bg-purple-50 p-2 rounded-lg text-purple-600"><Waves size={18} /></div>
                </div>
                <h3 className="text-3xl font-black text-slate-800">2<span className="text-lg font-bold text-slate-500 ml-1">건</span></h3>
              </div>
              <div className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex justify-between items-start mb-2">
                  <p className="text-sm font-bold text-slate-500">지게차 이상 (사각지대)</p>
                  <div className="bg-indigo-50 p-2 rounded-lg text-indigo-600"><Camera size={18} /></div>
                </div>
                <h3 className="text-3xl font-black text-slate-800">1<span className="text-lg font-bold text-slate-500 ml-1">건</span></h3>
              </div>
              <div className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex justify-between items-start mb-2">
                  <p className="text-sm font-bold text-slate-500">통신/센서 가동률</p>
                  <div className="bg-emerald-50 p-2 rounded-lg text-emerald-500"><Activity size={18} /></div>
                </div>
                <h3 className="text-3xl font-black text-slate-800">99.8<span className="text-lg font-bold text-slate-500 ml-1">%</span></h3>
              </div>
            </div>

            {/* 중간 2단 레이아웃: 작업자 프로필 & 시간대별 위험 추이 그래프 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* 차트 영역 (단순 CSS Bar 활용) */}
              <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm">
                <h3 className="text-lg font-bold text-slate-900 mb-6 flex items-center gap-2">
                  <Activity size={18} className="text-blue-500" /> 
                  시간대별 위험 추이
                </h3>
                <div className="relative h-48 border-b border-slate-200">
                  <div className="absolute inset-0 flex items-end justify-between px-2">
                    {/* 임의 시간대별 데이터 표시 (CSS 바) */}
                    {[10, 30, 15, 60, 20, 45, 10, 5, 25, 10].map((val, i) => (
                      <div key={i} className="w-1/12 flex flex-col justify-end items-center group">
                        <div className="w-full max-w-[24px] bg-blue-100 hover:bg-blue-400 rounded-t-md transition-colors relative" style={{ height: `${val}%` }}>
                          <span className="absolute -top-7 left-1/2 -translate-x-1/2 bg-slate-800 text-white text-[10px] font-bold px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity">
                            {val}건
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="flex justify-between px-2 mt-3 text-xs font-bold text-slate-400">
                  <span>09:00</span>
                  <span>12:00</span>
                  <span>15:00</span>
                  <span>18:00</span>
                </div>
              </div>

              {/* 작업자 프로파일 리스트 */}
              <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm">
                 <h3 className="text-lg font-bold text-slate-900 mb-6 flex items-center gap-2">
                  <Users size={18} className="text-blue-500" /> 
                  작업자 안전 스코어
                </h3>
                <div className="space-y-4">
                  {mockProfiles.map(profile => (
                    <div key={profile.id} className="flex items-center justify-between p-4 rounded-xl border border-slate-100 bg-slate-50/50 hover:bg-white hover:shadow-sm transition-all">
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 bg-slate-200 rounded-full flex items-center justify-center overflow-hidden border-2 border-white shadow-sm">
                          <img src={`https://api.dicebear.com/7.x/notionists/svg?seed=${profile.name}`} alt="avatar" className="w-full h-full object-cover" />
                        </div>
                        <div>
                          <p className="font-bold text-slate-900">{profile.name}</p>
                          <p className="text-xs font-semibold text-slate-500">{profile.role}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-6">
                        <div className="hidden sm:block text-right">
                          <p className="text-[10px] font-bold text-slate-400 mb-1 uppercase tracking-wider">안전도 추이</p>
                          <Sparkline data={profile.trend} />
                        </div>
                        <div className={`w-12 h-12 flex flex-col items-center justify-center rounded-xl font-bold ${
                          profile.score >= 90 ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-700'
                        }`}>
                          <span className="text-lg leading-none">{profile.score}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* 모달 연동 통합 위험 로그 테이블 */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
              <div className="p-6 border-b border-slate-100 flex justify-between items-center">
                <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                  <Video size={18} className="text-blue-500" />
                  멀티모달 통합 위험 관제 로그
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-slate-50/80 text-slate-500 text-xs font-bold tracking-wider uppercase border-b border-slate-200">
                      <th className="py-4 px-6">발생 정보</th>
                      <th className="py-4 px-6">작업 대상자</th>
                      <th className="py-4 px-6">CCTV / 오디오 증거</th>
                      <th className="py-4 px-6">상세 분석 내용</th>
                      <th className="py-4 px-6 text-center">액션</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {mockEvents.map((event) => (
                      <tr key={event.id} className="hover:bg-blue-50/30 transition-colors">
                        {/* 1. 시간 및 장비 */}
                        <td className="py-4 px-6">
                          <div className="font-bold text-slate-800 flex items-center gap-1.5 mb-1">
                            <Clock size={14} className="text-slate-400" /> {event.datetime}
                          </div>
                          <span className={`inline-flex px-2 py-0.5 rounded text-xs font-bold ${event.equipment === '크레인' ? 'bg-purple-100 text-purple-700' : 'bg-indigo-100 text-indigo-700'}`}>
                            {event.equipment}
                          </span>
                        </td>
                        
                        {/* 2. 작업자 */}
                        <td className="py-4 px-6 font-bold text-slate-700">
                           {event.worker}
                        </td>

                        {/* 3. 모달리티 썸네일 */}
                        <td className="py-4 px-6">
                          <div className="w-24 h-14 rounded-lg bg-slate-900 border-2 border-slate-800 flex items-center justify-center relative overflow-hidden group shadow-inner">
                            {event.type === '오디오' ? (
                              <div className="flex items-center gap-1">
                                <span className="w-1 h-3 bg-emerald-400 rounded-full animate-bounce" style={{animationDelay: '0s'}}></span>
                                <span className="w-1 h-6 bg-emerald-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></span>
                                <span className="w-1 h-4 bg-emerald-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></span>
                                <span className="w-1 h-8 bg-emerald-400 rounded-full animate-bounce" style={{animationDelay: '0.3s'}}></span>
                              </div>
                            ) : (
                              <>
                                <img src="https://images.unsplash.com/photo-1541888086425-d81bb19240f5?q=80&w=200&h=100&fit=crop" alt="cctv" className="absolute inset-0 w-full h-full object-cover opacity-60 mix-blend-luminosity group-hover:scale-110 transition-transform duration-500" />
                                <Camera size={18} className="text-white/80 absolute" />
                                <div className="absolute top-1 right-1 w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>
                              </>
                            )}
                          </div>
                        </td>

                        {/* 4. 분석 내용 */}
                        <td className="py-4 px-6">
                          <p className="text-sm font-bold text-red-600 mb-1">{event.analysis.split(']')[0]}]</p>
                          <p className="text-xs font-semibold text-slate-600">{event.analysis.split(']')[1]}</p>
                        </td>

                        {/* 5. 액션 버튼 */}
                        <td className="py-4 px-6 text-center">
                          <button 
                            onClick={() => setModalData({ isOpen: true, event })}
                            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 hover:border-blue-400 hover:text-blue-600 text-slate-600 text-xs font-bold rounded-lg shadow-sm transition-all"
                          >
                            <Eye size={14} /> 상세보기
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

          </div>
        </div>
      </main>

      {/* === Modal === */}
      {modalData.isOpen && modalData.event && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <div className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm" onClick={() => setModalData({ isOpen: false, event: null })}></div>
          {/* Modal Content */}
          <div className="relative bg-white rounded-2xl w-full max-w-xl p-0 overflow-hidden shadow-2xl animate-in fade-in zoom-in-95 duration-200">
            <div className="p-6 border-b border-slate-100 flex justify-between items-center bg-slate-50">
              <h3 className="font-bold text-slate-800 flex items-center gap-2">
                <AlertTriangle size={18} className="text-red-500" /> 
                비정상 상황 상세 리포트
              </h3>
              <button onClick={() => setModalData({ isOpen: false, event: null })} className="text-slate-400 hover:text-slate-700 bg-white p-1 rounded-md shadow-sm">
                ✕
              </button>
            </div>
            <div className="p-6 space-y-6">
              {/* 증거 미디어 표시 창 */}
              <div className="w-full h-48 bg-slate-900 rounded-xl flex items-center justify-center relative overflow-hidden shadow-inner">
                 {modalData.event.type === '오디오' ? (
                    <div className="text-center">
                      <Waves size={40} className="text-emerald-400 mx-auto mb-3" />
                      <p className="text-white text-sm font-bold tracking-widest">AUDIO RECORDING PLAYING...</p>
                    </div>
                  ) : (
                    <>
                      <img src="https://images.unsplash.com/photo-1541888086425-d81bb19240f5?q=80&w=800&h=400&fit=crop" alt="cctv full" className="w-full h-full object-cover opacity-80" />
                      <div className="absolute inset-0 border-[3px] border-red-500/50 rounded-xl pointer-events-none"></div>
                      <div className="absolute top-4 left-4 bg-red-600 text-white text-xs font-bold px-2 py-1 rounded">REC • CH-03</div>
                    </>
                  )}
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
                  <p className="text-xs text-slate-500 font-bold mb-1">발생 시각</p>
                  <p className="font-semibold text-slate-800">{modalData.event.datetime}</p>
                </div>
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
                  <p className="text-xs text-slate-500 font-bold mb-1">탐지 모달리티 / 장비</p>
                  <p className="font-semibold text-slate-800">{modalData.event.type} 기반 / {modalData.event.equipment}</p>
                </div>
              </div>

              <div>
                <p className="text-xs text-slate-500 font-bold mb-1">AI 분석 내용</p>
                <div className="bg-red-50 border border-red-100 p-3 rounded-lg text-red-700 text-sm font-bold">
                  {modalData.event.analysis}
                </div>
              </div>

              <div>
                <p className="text-xs text-slate-500 font-bold mb-1">안전 조치 내역</p>
                <div className="bg-blue-50 border border-blue-100 p-3 rounded-lg text-blue-700 text-sm font-bold flex items-center gap-2">
                  <CheckCircle size={16} /> {modalData.event.action}
                </div>
              </div>
            </div>
            <div className="p-4 bg-slate-50 border-t border-slate-100 text-right">
               <button onClick={() => setModalData({ isOpen: false, event: null })} className="bg-slate-800 text-white px-5 py-2.5 rounded-lg text-sm font-bold shadow-md hover:bg-slate-700 transition-colors">
                 확인 및 닫기
               </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
