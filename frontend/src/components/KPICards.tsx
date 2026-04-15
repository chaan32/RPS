import React from 'react';
import { Eye, ShieldAlert, Users, Award, ArrowUpRight, ArrowDownRight } from 'lucide-react';

export default function KPICards({ avgScore, totalPenaltyCount, dangerWorkers, safeDays }) {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4 2xl:gap-7.5">
      
      {/* Card 1: 전체 평균 안전 점수 */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-50 mb-5 border border-blue-100">
          <Eye className="text-blue-600 w-5 h-5" />
        </div>
        <div className="flex items-end justify-between">
          <div>
            <h4 className="text-3xl font-extrabold text-slate-800 tracking-tight">{avgScore}<span className="text-base text-slate-500 font-medium ml-1">점</span></h4>
            <span className="text-sm font-semibold text-slate-500 mt-1 block">전체 평균 안전 점수</span>
          </div>
          <span className="flex items-center gap-1 text-sm font-bold text-red-500 bg-red-50 px-2 py-0.5 rounded border border-red-100">
            -2.4 <ArrowDownRight className="w-3.5 h-3.5" />
          </span>
        </div>
      </div>

      {/* Card 2: 오늘 발생 총 감점 건수 */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-amber-50 mb-5 border border-amber-100">
          <ShieldAlert className="text-amber-500 w-5 h-5" />
        </div>
        <div className="flex items-end justify-between">
          <div>
            <h4 className="text-3xl font-extrabold text-slate-800 tracking-tight">{totalPenaltyCount}<span className="text-base text-slate-500 font-medium ml-1">건</span></h4>
            <span className="text-sm font-semibold text-slate-500 mt-1 block">오늘 발생 감점 건수</span>
          </div>
          <span className="flex items-center gap-1 text-sm font-bold text-red-500 bg-red-50 px-2 py-0.5 rounded border border-red-100">
            +3 <ArrowUpRight className="w-3.5 h-3.5" />
          </span>
        </div>
      </div>

      {/* Card 3: 위험군 인원 (High Visual Hierarchy) */}
      <div className="rounded-xl border border-red-200 bg-red-50 p-6 shadow-sm">
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-red-100 mb-5 border border-red-200">
          <Users className="text-red-600 w-5 h-5" />
        </div>
        <div className="flex items-end justify-between">
          <div>
            <h4 className="text-3xl font-extrabold text-red-600 tracking-tight">{dangerWorkers}<span className="text-base text-red-400 font-medium ml-1">명</span></h4>
            <span className="text-sm font-bold text-red-600/80 mt-1 block">현재 위험군 (70점 미만)</span>
          </div>
          <span className="flex items-center gap-1 text-sm font-bold text-red-500 bg-white px-2 py-0.5 rounded border border-red-100 shadow-sm">
            -1 <ArrowDownRight className="w-3.5 h-3.5" />
          </span>
        </div>
      </div>

      {/* Card 4: 무재해 일수 */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-emerald-50 mb-5 border border-emerald-100">
          <Award className="text-emerald-500 w-5 h-5" />
        </div>
        <div className="flex items-end justify-between">
          <div>
            <h4 className="text-3xl font-extrabold text-slate-800 tracking-tight">{safeDays}<span className="text-base text-slate-500 font-medium ml-1">일</span></h4>
            <span className="text-sm font-semibold text-slate-500 mt-1 block">연속 무재해 가동 기록</span>
          </div>
          <div className="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden mt-1 self-center">
             <div className="h-full bg-emerald-500 w-3/4"></div>
          </div>
        </div>
      </div>

    </div>
  );
}
