import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function TrendChart({ trendData }: { trendData: any[] }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-5 pt-7 pb-5 shadow-sm h-full flex flex-col">
       <div className="mb-4 flex flex-wrap justify-between gap-4">
         <div>
            <h4 className="text-xl font-bold text-slate-800 tracking-tight">시간대별 장비 감점 발생 추이</h4>
            <p className="text-sm font-medium text-slate-500 mt-1">현장의 크레인과 지게차 감점 발생 현황을 비교합니다.</p>
         </div>
       </div>

       <div className="h-80 w-full mt-4 -ml-4 flex-grow">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={trendData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="colorForklift" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4}/>
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="colorCrane" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.4}/>
                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
            <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748b', fontWeight: 500 }} dy={10} minTickGap={20} />
            <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748b', fontWeight: 500 }} />
            <Tooltip 
              contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
              itemStyle={{ fontSize: '13px', fontWeight: 600, color: '#0f172a' }}
            />
            <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '12px', fontWeight: 600, color: '#475569' }} />
            {/* type선택시 linear를 사용하여 뾰족한 차트 느낌을 살림 */}
            <Area type="linear" name="지게차 감점" dataKey="forklift" stroke="#3b82f6" strokeWidth={3} fillOpacity={1} fill="url(#colorForklift)" activeDot={{ r: 6, strokeWidth: 0, fill: '#2563eb' }} />
            <Area type="linear" name="크레인 감점" dataKey="crane" stroke="#8b5cf6" strokeWidth={3} fillOpacity={1} fill="url(#colorCrane)" activeDot={{ r: 6, strokeWidth: 0, fill: '#7c3aed' }} />
          </AreaChart>
        </ResponsiveContainer>
       </div>
    </div>
  )
}
