import { useState } from 'react';
import { 
  Menu, Search, Bell, MessageSquare, LayoutDashboard, Calendar, User, 
  ChevronDown, Activity, Calendar as CalendarIcon, Download
} from 'lucide-react';
import KPICards from './KPICards';
import TrendChart from './TrendChart';
import RiskFactorDonut from './RiskFactorDonut';
import WorkerList from './WorkerList';
import WorkerProfileModal from './WorkerProfileModal';
import { nodeScores, trendData, events } from '../data/mockData';

export default function SafetyDashboard() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedWorker, setSelectedWorker] = useState<any>(null);

  // 상단 요약 데이터 계산
  const avgScore = (nodeScores.reduce((acc, curr) => acc + curr.finalScore, 0) / nodeScores.length).toFixed(1);
  const totalPenaltyCount = events.length;
  const dangerWorkers = nodeScores.filter(n => n.finalScore < 70).length;
  const safeDays = 14;

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50 font-sans text-slate-800 flex-col md:flex-row">
      {/* Sidebar */}
      <aside className={`absolute left-0 top-0 z-40 flex h-screen w-64 flex-col bg-[#1c2434] duration-300 ease-linear md:static md:translate-x-0 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="flex items-center gap-2 px-6 py-6 border-b border-white/10">
          <div className="bg-blue-600 p-1.5 rounded text-white flex items-center justify-center shadow-lg shadow-blue-600/20">
             <Activity className="w-5 h-5" />
          </div>
          <span className="text-white text-xl font-bold tracking-wide">Safe<span className="text-blue-500">Node</span></span>
        </div>
        <div className="flex-1 overflow-y-auto no-scrollbar">
          <nav className="mt-6 px-4">
            <h3 className="mb-4 ml-4 text-xs font-semibold text-[#8a99af] tracking-widest uppercase">Menu</h3>
            <ul className="flex flex-col gap-1.5">
              <li>
                <a href="#" className="flex items-center gap-3 rounded-md bg-[#333a48] py-2.5 px-4 font-medium text-white transition-all shadow-sm">
                  <LayoutDashboard className="w-5 h-5 text-slate-300" /> Dashboard
                </a>
              </li>
              <li>
                <a href="#" className="flex items-center gap-3 rounded-md py-2.5 px-4 font-medium text-[#8a99af] hover:bg-[#333a48] hover:text-white transition-colors">
                  <Calendar className="w-5 h-5" /> Calendar
                </a>
              </li>
              <li>
                <a href="#" className="flex items-center gap-3 rounded-md py-2.5 px-4 font-medium text-[#8a99af] hover:bg-[#333a48] hover:text-white transition-colors">
                  <User className="w-5 h-5" /> Profile
                </a>
              </li>
            </ul>
          </nav>
        </div>
      </aside>

      {/* Main Area */}
      <div className="relative flex flex-1 flex-col overflow-y-auto overflow-x-hidden">
        {/* Header */}
        <header className="sticky top-0 z-30 flex w-full bg-white shadow-sm border-b border-slate-200">
          <div className="flex flex-grow items-center justify-between px-4 py-3 md:px-6">
            <div className="flex items-center gap-2 md:hidden">
              <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-1.5 border border-slate-200 rounded-md bg-white">
                <Menu className="w-5 h-5 text-slate-500" />
              </button>
            </div>
            
            <div className="hidden sm:block">
              <div className="relative">
                <Search className="absolute left-0 w-4.5 h-4.5 text-slate-400 top-1/2 -translate-y-1/2" />
                <input type="text" placeholder="Type to search..." className="bg-transparent pl-8 pr-4 text-sm focus:outline-none w-64 text-slate-700 font-medium" />
              </div>
            </div>

            <div className="flex items-center gap-3 2xsm:gap-7 ml-auto">
              {/* 추가된 Global Controls */}
              <div className="hidden xl:flex items-center gap-2 mr-2">
                <button className="flex items-center gap-2 px-3 py-2 rounded-lg border border-slate-200 bg-slate-50 text-slate-600 text-[13px] font-bold hover:bg-slate-100 transition-colors">
                  <CalendarIcon className="w-4 h-4" />
                  오늘 (2026. 04. 14)
                </button>
                <button className="flex items-center gap-2 px-3 py-2 rounded-lg bg-blue-600 text-white text-[13px] font-bold hover:bg-blue-700 transition-colors shadow-sm shadow-blue-600/20">
                  <Download className="w-4 h-4" />
                  보고서 다운로드 (PDF)
                </button>
              </div>

              <ul className="flex items-center gap-2">
                <li>
                  <button className="relative p-2 bg-slate-50 border border-slate-100 rounded-full text-slate-500 hover:text-blue-600 transition-colors">
                    <MessageSquare className="w-4.5 h-4.5" />
                    <span className="absolute top-1 right-1.5 h-2 w-2 rounded-full bg-red-500 border border-white"></span>
                  </button>
                </li>
                <li>
                  <button className="relative p-2 bg-slate-50 border border-slate-100 rounded-full text-slate-500 hover:text-blue-600 transition-colors">
                    <Bell className="w-4.5 h-4.5" />
                    <span className="absolute top-1 right-1.5 h-2 w-2 rounded-full bg-red-500 border border-white"></span>
                  </button>
                </li>
              </ul>
              
              <div className="flex items-center gap-3 pl-4 border-l border-slate-200">
                <div className="text-right hidden lg:block">
                  <span className="block text-sm font-bold text-slate-800">Thomas Anree</span>
                  <span className="block text-xs text-slate-500 font-medium whitespace-nowrap">Safety Manager</span>
                </div>
                <div className="h-10 w-10 bg-slate-200 rounded-full overflow-hidden border-2 border-slate-100 shadow-sm">
                  <img src="https://i.pravatar.cc/150?img=11" alt="Profile" className="h-full w-full object-cover" />
                </div>
                <ChevronDown className="hidden w-4 h-4 text-slate-400 sm:block" />
              </div>
            </div>
          </div>
        </header>

        {/* Dashboard Content */}
        <main className="p-4 md:p-6 2xl:p-8 flex-grow">
          <KPICards 
            avgScore={avgScore} 
            totalPenaltyCount={totalPenaltyCount} 
            dangerWorkers={dangerWorkers} 
            safeDays={safeDays} 
          />

          {/* 7:3 레이아웃 분할된 차트 영역 */}
          <div className="mt-4 grid grid-cols-1 gap-4 md:mt-6 md:gap-6 2xl:mt-7.5 2xl:gap-7.5 lg:grid-cols-12">
            <div className="lg:col-span-8 2xl:col-span-8">
               <TrendChart trendData={trendData} />
            </div>
            <div className="lg:col-span-4 2xl:col-span-4">
               <RiskFactorDonut />
            </div>
          </div>

          <div className="mt-4 grid grid-cols-1 gap-4 md:mt-6 md:gap-6 2xl:mt-7.5 2xl:gap-7.5">
            <div className="col-span-1">
               <WorkerList 
                  nodeScores={nodeScores} 
                  onWorkerClick={setSelectedWorker} 
               />
            </div>
          </div>
        </main>
      </div>

      {selectedWorker && (
        <WorkerProfileModal 
          worker={selectedWorker} 
          onClose={() => setSelectedWorker(null)} 
        />
      )}
    </div>
  );
}
