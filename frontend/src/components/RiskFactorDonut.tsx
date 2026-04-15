import React from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { riskFactors, RISK_COLORS } from '../data/mockData';

const RADIAN = Math.PI / 180;
const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }: any) => {
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);

  return percent > 0.05 ? (
    <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central" fontSize="11" fontWeight="bold">
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  ) : null;
};

export default function RiskFactorDonut() {
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-5 pt-7 pb-5 shadow-sm h-full flex flex-col">
       <div className="mb-4">
          <h4 className="text-xl font-bold text-slate-800 tracking-tight">주요 위험 발생 요인</h4>
          <p className="text-sm font-medium text-slate-500 mt-1">오늘 발생한 감점 요인별 비율입니다.</p>
       </div>
       <div className="h-64 w-full flex-grow mt-2">
          {/* 중앙의 100% 텍스트를 제거하고 Pie에 Label만 할당 */}
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
               <Pie
                 data={riskFactors}
                 cx="50%"
                 cy="50%"
                 innerRadius={"45%"}
                 outerRadius={"85%"}
                 paddingAngle={2}
                 dataKey="value"
                 stroke="none"
                 labelLine={false}
                 label={renderCustomizedLabel}
               >
                 {riskFactors.map((entry, index) => (
                   <Cell key={`cell-${index}`} fill={RISK_COLORS[index % RISK_COLORS.length]} />
                 ))}
               </Pie>
               <Tooltip 
                 contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                 itemStyle={{ fontSize: '13px', fontWeight: 600, color: '#0f172a' }}
                 formatter={(value: any) => [`${value}%`, '차지 비율']}
               />
               <Legend 
                 layout="horizontal" 
                 verticalAlign="bottom" 
                 align="center"
                 iconType="circle"
                 wrapperStyle={{ fontSize: '12px', fontWeight: 600, color: '#475569', marginTop: '10px' }}
               />
            </PieChart>
          </ResponsiveContainer>
       </div>
    </div>
  )
}
