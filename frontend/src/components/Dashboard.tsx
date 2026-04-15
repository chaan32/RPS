import React, { useState } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, Legend
} from 'recharts';
import {
  Menu, Search, Bell, MessageSquare, LayoutDashboard, Calendar, User,
  CheckSquare, FileText, Table, Settings, AlertTriangle, CheckCircle,
  Activity, MapPin, Clock, ArrowUpRight, ArrowDownRight, Truck, Box, ChevronDown,
  Eye, ShoppingCart, ShoppingBag, Users
} from 'lucide-react';

// ==========================================
// 1. Mock Data
// ==========================================

const MOCK_HOURLY = [
  { time: '09:00', crane: 12, forklift: 24 },
  { time: '10:00', crane: 15, forklift: 20 },
  { time: '11:00', crane: 18, forklift: 35 },
  { time: '12:00', crane: 5, forklift: 10 },
  { time: '13:00', crane: 14, forklift: 30 },
  { time: '14:00', crane: 25, forklift: 45 },
  { time: '15:00', crane: 20, forklift: 30 },
  { time: '16:00', crane: 15, forklift: 20 },
  { time: '17:00', crane: 10, forklift: 35 },
  { time: '18:00', crane: 5, forklift: 15 },
];

const MOCK_DONUT = [
  { name: '크레인 근접경고', value: 35 },
  { name: '지게차 속도초과', value: 50 },
  { name: '기타 이벤트', value: 15 },
];

const PIE_COLORS = ['#3b82f6', '#60a5fa', '#bfdbfe'];

const MOCK_BAR = [
  { zone: 'A창고', events: 45 },
  { zone: 'B하역장', events: 65 },
  { zone: 'C조립', events: 30 },
  { zone: '야적장', events: 50 },
  { zone: '검수', events: 20 },
  { zone: '출하장', events: 35 },
  { zone: '설비동', events: 15 },
];

const TIMELINE_EVENTS = [
  { id: 1, time: '17:42', equip: '지게차 #4', zone: 'B하역장', severity: 'danger', detail: '제한속도 15km/h 초과 (20km/h)' },
  { id: 2, time: '16:30', equip: '크레인 #2', zone: 'A창고', severity: 'warning', detail: '작업자 2m 이내 근접 감지' },
  { id: 3, time: '15:15', equip: '크레인 #1', zone: 'A창고', severity: 'warning', detail: '적재함 정격 하중 95% 임박' },
  { id: 4, time: '14:20', equip: '지게차 #1', zone: 'C조립라인', severity: 'danger', detail: '교차로 보행자 충돌 위험 (긴급제동)' },
  { id: 5, time: '13:05', equip: '센서시스템', zone: '야적장', severity: 'safe', detail: '통신 모듈 재연결 성공 (정상)' },
];

// ==========================================
// 2. Dashboard Component
// ==========================================

export default function TailAdminSafetyDashboard() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50 font-sans text-slate-800">
      {/* Sidebar */}
      <aside className={`absolute left-0 top-0 z-40 flex h-screen w-64 flex-col overflow-y-hidden bg-[#1c2434] duration-300 ease-linear lg:static lg:translate-x-0 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="flex items-center gap-2 px-6 py-6 border-b border-white/10">
          <div className="bg-blue-600 p-1.5 rounded text-white flex items-center justify-center">
            <Activity className="w-5 h-5" />
          </div>
          <span className="text-white text-xl font-bold tracking-wide">TailAdmin<span className="text-blue-500">.</span></span>
        </div>

        <div className="no-scrollbar flex flex-col overflow-y-auto">
          <nav className="mt-6 px-4">
            <h3 className="mb-4 ml-4 text-xs font-semibold text-[#8a99af] tracking-widest uppercase">Menu</h3>
            <ul className="flex flex-col gap-1.5">
              <li>
                <a href="#" className="group relative flex items-center gap-3 rounded-md bg-[#333a48] py-2.5 px-4 font-medium text-white transition-all">
                  <LayoutDashboard className="w-5 h-5 text-slate-300" />
                  Dashboard
                </a>
              </li>
              <li>
                <a href="#" className="group relative flex items-center gap-3 rounded-md py-2.5 px-4 font-medium text-[#8a99af] hover:bg-[#333a48] hover:text-white transition-colors">
                  <Calendar className="w-5 h-5" />
                  Calendar
                </a>
              </li>
              <li>
                <a href="#" className="group relative flex items-center gap-3 rounded-md py-2.5 px-4 font-medium text-[#8a99af] hover:bg-[#333a48] hover:text-white transition-colors">
                  <User className="w-5 h-5" />
                  Profile
                </a>
              </li>
              <li>
                <a href="#" className="group relative flex items-center gap-3 rounded-md py-2.5 px-4 font-medium text-[#8a99af] hover:bg-[#333a48] hover:text-white transition-colors">
                  <CheckSquare className="w-5 h-5" />
                  Task
                </a>
              </li>
              <li>
                <a href="#" className="group relative flex items-center gap-3 rounded-md py-2.5 px-4 font-medium text-[#8a99af] hover:bg-[#333a48] hover:text-white transition-colors">
                  <FileText className="w-5 h-5" />
                  Forms
                </a>
              </li>
              <li>
                <a href="#" className="group relative flex items-center gap-3 rounded-md py-2.5 px-4 font-medium text-[#8a99af] hover:bg-[#333a48] hover:text-white transition-colors">
                  <Table className="w-5 h-5" />
                  Tables
                </a>
              </li>
              <li>
                <a href="#" className="group relative flex items-center gap-3 rounded-md py-2.5 px-4 font-medium text-[#8a99af] hover:bg-[#333a48] hover:text-white transition-colors">
                  <Settings className="w-5 h-5" />
                  Settings
                </a>
              </li>
            </ul>

            <h3 className="mb-4 ml-4 mt-8 text-xs font-semibold text-[#8a99af] tracking-widest uppercase">Support</h3>
            <ul className="flex flex-col gap-1.5">
              <li>
                <a href="#" className="group relative flex items-center justify-between rounded-md py-2.5 px-4 font-medium text-[#8a99af] hover:bg-[#333a48] hover:text-white transition-colors">
                  <div className="flex items-center gap-3">
                    <MessageSquare className="w-5 h-5" />
                    Messages
                  </div>
                  <span className="bg-blue-600 text-white text-[10px] font-bold rounded px-2 py-0.5">5</span>
                </a>
              </li>
              <li>
                <a href="#" className="group relative flex items-center justify-between rounded-md py-2.5 px-4 font-medium text-[#8a99af] hover:bg-[#333a48] hover:text-white transition-colors">
                  <div className="flex items-center gap-3">
                    <FileText className="w-5 h-5" />
                    Inbox
                  </div>
                  <span className="bg-blue-600 text-white text-[10px] font-bold rounded px-2 py-0.5">Pro</span>
                </a>
              </li>
              <li>
                <a href="#" className="group relative flex items-center justify-between rounded-md py-2.5 px-4 font-medium text-[#8a99af] hover:bg-[#333a48] hover:text-white transition-colors">
                  <div className="flex items-center gap-3">
                    <Table className="w-5 h-5" />
                    Invoice
                  </div>
                  <span className="bg-blue-600 text-white text-[10px] font-bold rounded px-2 py-0.5">Pro</span>
                </a>
              </li>
            </ul>
          </nav>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="relative flex flex-1 flex-col overflow-y-auto overflow-x-hidden">
        {/* Top Header */}
        <header className="sticky top-0 z-30 flex w-full bg-white drop-shadow-sm border-b border-slate-200">
          <div className="flex flex-grow items-center justify-between px-4 py-3 md:px-6 2xl:px-8">
            {/* Mobile Sidebar Toggle & Search */}
            <div className="flex items-center gap-2 sm:gap-4 w-full">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="z-50 block rounded-md border border-slate-200 bg-white p-1.5 shadow-sm lg:hidden focus:ring-2 focus:ring-blue-500"
              >
                <Menu className="w-5 h-5 text-slate-500" />
              </button>

              <div className="hidden sm:block w-full max-w-md">
                <div className="relative flex items-center">
                  <Search className="absolute left-0 w-4.5 h-4.5 text-slate-400" />
                  <input
                    type="text"
                    placeholder="Type to search..."
                    className="w-full bg-transparent pl-8 pr-4 text-sm focus:outline-none text-slate-700 font-medium"
                  />
                </div>
              </div>
            </div>

            {/* Right Header Area */}
            <div className="flex items-center gap-3 2xsm:gap-7 shrink-0">
              <ul className="flex items-center gap-2">
                <li>
                  <button className="relative flex h-8.5 w-8.5 items-center justify-center rounded-full bg-slate-100 hover:text-blue-600 p-2 text-slate-500 transition-colors">
                    <span className="absolute top-1 right-1.5 z-1 h-2 w-2 rounded-full bg-red-500 border border-white"></span>
                    <MessageSquare className="w-4.5 h-4.5" />
                  </button>
                </li>
                <li>
                  <button className="relative flex h-8.5 w-8.5 items-center justify-center rounded-full bg-slate-100 hover:text-blue-600 p-2 text-slate-500 transition-colors">
                    <span className="absolute top-1 right-2 z-1 h-2 w-2 rounded-full bg-red-500 border border-white"></span>
                    <Bell className="w-4.5 h-4.5" />
                  </button>
                </li>
              </ul>

              <div className="flex items-center gap-3 pl-4 border-l border-slate-200">
                <span className="hidden text-right lg:block">
                  <span className="block text-sm font-semibold text-slate-800">Thomas Anree</span>
                  <span className="block text-xs text-slate-500 font-medium">Safety Director</span>
                </span>
                <div className="h-10 w-10 rounded-full bg-slate-200 overflow-hidden border border-slate-300">
                  <img src="https://i.pravatar.cc/150?img=11" alt="User" className="h-full w-full object-cover" />
                </div>
                <ChevronDown className="hidden w-4 h-4 text-slate-400 sm:block" />
              </div>
            </div>
          </div>
        </header>

        {/* Dashboard Content */}
        <main className="p-4 md:p-6 2xl:p-8 flex-grow">

          {/* Top KPI Cards (Top 4) */}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4 2xl:gap-7.5">

            {/* Card 1: 총 안전 이벤트 */}
            <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-[0_2px_10px_-3px_rgba(6,81,237,0.1)]">
              <div className="flex h-11.5 w-11.5 items-center justify-center rounded-full bg-blue-50 w-12 h-12 mb-5">
                <Eye className="text-blue-600 w-5 h-5" />
              </div>
              <div className="flex items-end justify-between">
                <div>
                  <h4 className="text-title-md font-bold text-slate-800 text-3xl tracking-tight">452</h4>
                  <span className="text-sm font-semibold text-slate-500 mt-1 block">총 안전 이벤트</span>
                </div>
                <span className="flex items-center gap-1 text-sm font-bold text-emerald-500 bg-emerald-50 px-2 py-0.5 rounded">
                  0.43% <ArrowUpRight className="w-3.5 h-3.5" />
                </span>
              </div>
            </div>

            {/* Card 2: 크레인 근접 경고 */}
            <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-[0_2px_10px_-3px_rgba(6,81,237,0.1)]">
              <div className="flex h-11.5 w-11.5 items-center justify-center rounded-full bg-blue-50 w-12 h-12 mb-5">
                <ShoppingCart className="text-blue-600 w-5 h-5" /> {/* UI 벤치마킹을 위해 장바구니 아이콘 유지 */}
              </div>
              <div className="flex items-end justify-between">
                <div>
                  <h4 className="text-title-md font-bold text-slate-800 text-3xl tracking-tight">45</h4>
                  <span className="text-sm font-semibold text-slate-500 mt-1 block">크레인 근접 경고</span>
                </div>
                <span className="flex items-center gap-1 text-sm font-bold text-emerald-500 bg-emerald-50 px-2 py-0.5 rounded">
                  4.35% <ArrowUpRight className="w-3.5 h-3.5" />
                </span>
              </div>
            </div>

            {/* Card 3: 지게차 속도 초과 */}
            <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-[0_2px_10px_-3px_rgba(6,81,237,0.1)]">
              <div className="flex h-11.5 w-11.5 items-center justify-center rounded-full bg-blue-50 w-12 h-12 mb-5">
                <ShoppingBag className="text-blue-600 w-5 h-5" />
              </div>
              <div className="flex items-end justify-between">
                <div>
                  <h4 className="text-title-md font-bold text-slate-800 text-3xl tracking-tight">128</h4>
                  <span className="text-sm font-semibold text-slate-500 mt-1 block">지게차 속도 초과</span>
                </div>
                <span className="flex items-center gap-1 text-sm font-bold text-emerald-500 bg-emerald-50 px-2 py-0.5 rounded">
                  2.59% <ArrowUpRight className="w-3.5 h-3.5" />
                </span>
              </div>
            </div>

            {/* Card 4: 정상 가동 센서 */}
            <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-[0_2px_10px_-3px_rgba(6,81,237,0.1)]">
              <div className="flex h-11.5 w-11.5 items-center justify-center rounded-full bg-blue-50 w-12 h-12 mb-5">
                <Users className="text-blue-600 w-5 h-5" />
              </div>
              <div className="flex items-end justify-between">
                <div>
                  <h4 className="text-title-md font-bold text-slate-800 text-3xl tracking-tight">99.8<span className="text-xl">%</span></h4>
                  <span className="text-sm font-semibold text-slate-500 mt-1 block">정상 가동 센서</span>
                </div>
                <span className="flex items-center gap-1 text-sm font-bold text-blue-500 bg-blue-50 px-2 py-0.5 rounded">
                  0.95% <ArrowDownRight className="w-3.5 h-3.5" />
                </span>
              </div>
            </div>
          </div>

          {/* Middle Charts Row */}
          <div className="mt-4 grid grid-cols-12 gap-4 md:mt-6 md:gap-6 2xl:mt-7.5 2xl:gap-7.5">

            {/* Area Chart: 시간대별 위험 이벤트 발생 추이 */}
            <div className="col-span-12 rounded-xl border border-slate-200 bg-white px-5 pt-7 pb-5 shadow-[0_2px_10px_-3px_rgba(6,81,237,0.1)] xl:col-span-8">
              <div className="flex flex-wrap items-start justify-between gap-3 sm:flex-nowrap">
                <div className="flex w-full flex-wrap gap-3 sm:gap-5">
                  <div className="flex min-w-47.5">
                    <span className="mt-1 mr-2 flex h-4 w-4 items-center justify-center rounded-full border border-blue-600">
                      <span className="block h-2.5 w-2.5 rounded-full bg-blue-600"></span>
                    </span>
                    <div className="w-full">
                      <p className="font-semibold text-blue-600">크레인 이벤트</p>
                      <p className="text-xs font-semibold text-slate-500 mt-0.5">09:00 - 18:00</p>
                    </div>
                  </div>
                  <div className="flex min-w-47.5">
                    <span className="mt-1 mr-2 flex h-4 w-4 items-center justify-center rounded-full border border-blue-300">
                      <span className="block h-2.5 w-2.5 rounded-full bg-blue-400"></span>
                    </span>
                    <div className="w-full">
                      <p className="font-semibold text-blue-400">지게차 이벤트</p>
                      <p className="text-xs font-semibold text-slate-500 mt-0.5">09:00 - 18:00</p>
                    </div>
                  </div>
                </div>
                <div className="flex w-full max-w-45 justify-end">
                  <div className="inline-flex items-center rounded-md bg-slate-100 p-1">
                    <button className="rounded py-1 px-3 text-xs font-semibold bg-white text-black shadow-sm transition">Day</button>
                    <button className="rounded py-1 px-3 text-xs font-semibold text-slate-500 hover:text-black transition">Week</button>
                    <button className="rounded py-1 px-3 text-xs font-semibold text-slate-500 hover:text-black transition">Month</button>
                  </div>
                </div>
              </div>

              <div className="h-[320px] w-full mt-6 -ml-4">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={MOCK_HOURLY} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="colorCrane" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.35} />
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="colorForklift" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#60a5fa" stopOpacity={0.15} />
                        <stop offset="95%" stopColor="#60a5fa" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                    <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748b', fontWeight: 500 }} dy={10} />
                    <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748b', fontWeight: 500 }} />
                    <Tooltip
                      contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                      itemStyle={{ fontSize: '13px', fontWeight: 600 }}
                    />
                    <Area type="monotone" dataKey="crane" stroke="#3b82f6" strokeWidth={2.5} fillOpacity={1} fill="url(#colorCrane)" activeDot={{ r: 5, strokeWidth: 0 }} />
                    <Area type="monotone" dataKey="forklift" stroke="#60a5fa" strokeWidth={2.5} fillOpacity={1} fill="url(#colorForklift)" activeDot={{ r: 5, strokeWidth: 0 }} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Donut Chart: 장비별 비율 */}
            <div className="col-span-12 rounded-xl border border-slate-200 bg-white px-5 pt-7 pb-5 shadow-[0_2px_10px_-3px_rgba(6,81,237,0.1)] xl:col-span-4 flex flex-col">
              <div className="mb-4 flex justify-between items-center">
                <h4 className="text-lg font-bold text-slate-800">장비별 위험 비율</h4>
                <div className="relative z-20 inline-block bg-slate-50 border border-slate-100 rounded px-2">
                  <select className="relative z-20 outline-none inline-flex appearance-none bg-transparent py-1 pl-1 pr-6 text-sm font-semibold text-slate-600">
                    <option value="">Today</option>
                  </select>
                  <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
                </div>
              </div>

              <div className="flex-grow flex items-center justify-center w-full mt-4">
                <div className="h-64 w-full relative">
                  {/* Center Text inside Donut */}
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-center mt-[-10px] pointer-events-none">
                    <span className="text-3xl font-extrabold text-slate-800 block">100%</span>
                    <span className="text-[11px] text-slate-500 font-bold tracking-widest mt-1 block">TOTAL RATIO</span>
                  </div>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={MOCK_DONUT}
                        cx="50%"
                        cy="45%"
                        innerRadius={78}
                        outerRadius={95}
                        paddingAngle={2}
                        dataKey="value"
                        stroke="none"
                        cornerRadius={3}
                      >
                        {MOCK_DONUT.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                        itemStyle={{ fontSize: '13px', fontWeight: 600 }}
                      />
                      <Legend
                        layout="horizontal"
                        verticalAlign="bottom"
                        align="center"
                        iconType="circle"
                        wrapperStyle={{ fontSize: '13px', fontWeight: 600, color: '#475569', marginTop: '10px' }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Row */}
          <div className="mt-4 grid grid-cols-12 gap-4 md:mt-6 md:gap-6 2xl:mt-7.5 2xl:gap-7.5">

            {/* Bar Chart: 구역별 이벤트 현황 */}
            <div className="col-span-12 rounded-xl border border-slate-200 bg-white px-5 pt-7 pb-5 shadow-[0_2px_10px_-3px_rgba(6,81,237,0.1)] xl:col-span-6">
              <div className="mb-4 flex justify-between items-center">
                <h4 className="text-lg font-bold text-slate-800">구역별 위험 이벤트 건수</h4>
                <div className="relative z-20 inline-block bg-slate-50 border border-slate-100 rounded px-2">
                  <select className="relative z-20 outline-none inline-flex appearance-none bg-transparent py-1 pl-1 pr-6 text-sm font-semibold text-slate-600">
                    <option value="">Monthly</option>
                  </select>
                  <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
                </div>
              </div>
              <div className="h-80 w-full mt-4 -ml-4">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={MOCK_BAR} margin={{ top: 10, right: 10, left: 0, bottom: 0 }} barSize={16}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                    <XAxis dataKey="zone" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748b', fontWeight: 500 }} dy={10} />
                    <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748b', fontWeight: 500 }} />
                    <Tooltip
                      contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                      cursor={{ fill: '#f8fafc' }}
                    />
                    <Bar dataKey="events" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Timeline Log Table (대체) */}
            <div className="col-span-12 rounded-xl border border-slate-200 bg-white shadow-[0_2px_10px_-3px_rgba(6,81,237,0.1)] xl:col-span-6 flex flex-col">
              <div className="px-6 py-6 border-b border-slate-100 flex justify-between items-center">
                <h4 className="text-lg font-bold text-slate-800">최신 안전 이벤트 타임라인 로그</h4>
              </div>
              <div className="flex-grow overflow-x-auto">
                <table className="w-full text-left text-sm text-slate-700">
                  <thead className="bg-[#f7f9fc] text-slate-500 font-bold text-[11px] uppercase tracking-wider">
                    <tr>
                      <th className="px-6 py-4">시간</th>
                      <th className="px-6 py-4">구역</th>
                      <th className="px-6 py-4">장비/타입</th>
                      <th className="px-6 py-4">내용</th>
                      <th className="px-6 py-4">심각도</th>
                    </tr>
                  </thead>
                  <tbody>
                    {TIMELINE_EVENTS.map((event, i) => (
                      <tr key={event.id} className="border-b border-slate-100 last:border-0 hover:bg-slate-50/50 transition-colors">
                        <td className="px-6 py-4 whitespace-nowrap font-semibold text-slate-800">{event.time}</td>
                        <td className="px-6 py-4 font-medium text-slate-600">{event.zone}</td>
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2 font-bold text-slate-800">
                            {event.equip.includes('크레인') ? <Activity className="w-4 h-4 text-blue-500" /> : event.equip.includes('지게차') ? <Truck className="w-4 h-4 text-slate-400" /> : <Settings className="w-4 h-4 text-slate-400" />}
                            {event.equip}
                          </div>
                        </td>
                        <td className="px-6 py-4 font-medium text-slate-600 truncate max-w-[200px]">{event.detail}</td>
                        <td className="px-6 py-4">
                          {event.severity === 'danger' && <span className="inline-flex rounded-full bg-red-50 px-2.5 py-1 text-[11px] font-bold text-red-600 border border-red-100">Danger</span>}
                          {event.severity === 'warning' && <span className="inline-flex rounded-full bg-amber-50 px-2.5 py-1 text-[11px] font-bold text-amber-600 border border-amber-100">Warning</span>}
                          {event.severity === 'safe' && <span className="inline-flex rounded-full bg-emerald-50 px-2.5 py-1 text-[11px] font-bold text-emerald-600 border border-emerald-100">Safe</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

          </div>

        </main>
      </div>
    </div>
  );
}
