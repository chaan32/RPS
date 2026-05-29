import { useState } from 'react';
import { User, Search, Clock } from 'lucide-react';

export default function WorkerList({ nodeScores, onWorkerClick }: { nodeScores: any[], onWorkerClick: (w:any) => void }) {
  const [filter, setFilter] = useState('ALL');

  const filteredScores = nodeScores.filter(worker => {
    if (filter === 'ALL') return true;
    return worker.role === filter;
  });

  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden flex flex-col h-full mt-4">
       <div className="px-6 py-5 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
         <div>
           <h4 className="text-xl font-bold text-slate-800 tracking-tight">작업자 노드 상태 관리 그리드</h4>
           <p className="text-sm font-medium text-slate-500 mt-1">현장의 모든 작업자를 안전 점수 기준으로 모니터링합니다.</p>
         </div>
         <div className="flex bg-slate-100 p-1 rounded-lg self-start sm:self-auto">
            <button 
              onClick={() => setFilter('ALL')} 
              className={`px-4 py-1.5 rounded-md text-sm font-bold transition-all ${filter === 'ALL' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
            >전체보기</button>
            <button 
              onClick={() => setFilter('지게차')} 
              className={`px-4 py-1.5 rounded-md text-sm font-bold transition-all ${filter === '지게차' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
            >지게차</button>
            <button 
              onClick={() => setFilter('크레인')} 
              className={`px-4 py-1.5 rounded-md text-sm font-bold transition-all ${filter === '크레인' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
            >크레인</button>
         </div>
       </div>
       <div className="overflow-x-auto flex-grow bg-white">
          <table className="w-full text-left text-sm text-slate-700">
             <thead className="bg-[#f7f9fc] text-slate-500 font-bold text-[11px] uppercase tracking-wider">
               <tr>
                 <th className="px-6 py-4 border-b border-slate-200">근무자 ID</th>
                 <th className="px-6 py-4 border-b border-slate-200">이름</th>
                 <th className="px-6 py-4 border-b border-slate-200">직무 종류</th>
                 {/* 신규 추가된 '최근 발생 위반' 열 */}
                 <th className="px-6 py-4 border-b border-slate-200">최근 발생 위반 (Latest Event)</th>
                 <th className="px-6 py-4 border-b border-slate-200">임계 누적 감점</th>
                 <th className="px-6 py-4 border-b border-slate-200">최종 안전 점수</th>
                 <th className="px-6 py-4 border-b border-slate-200 text-right">상세/조치</th>
               </tr>
             </thead>
             <tbody>
               {filteredScores.map((worker) => {
                 const scoreColor = worker.finalScore >= 90 ? 'text-emerald-500' : worker.finalScore >= 70 ? 'text-amber-500' : 'text-red-500';
                 const badgeBg = worker.finalScore >= 90 ? 'bg-emerald-50 border-emerald-100' : worker.finalScore >= 70 ? 'bg-amber-50 border-amber-100' : 'bg-red-50 border-red-100';
                 
                 return (
                   <tr key={worker.id} onClick={() => onWorkerClick(worker)} className="border-b border-slate-100 last:border-0 hover:bg-slate-50 transition-colors cursor-pointer group">
                     <td className="px-6 py-4 font-mono font-semibold text-slate-500 group-hover:text-blue-600 transition-colors">{worker.id}</td>
                     <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className="bg-slate-100 p-1.5 rounded-full text-slate-500 group-hover:bg-blue-100 group-hover:text-blue-600 transition-colors">
                             <User className="w-4 h-4" />
                          </div>
                          <span className="font-bold text-slate-800 whitespace-nowrap">{worker.name}</span>
                        </div>
                     </td>
                     <td className="px-6 py-4 font-bold text-slate-600">{worker.role}</td>
                     
                     {/* 최근 발생 위반 데이터 렌더링 구역 */}
                     <td className="px-6 py-4">
                        {worker.latestEvent ? (
                          <div className="flex flex-col items-start gap-1">
                            <span className="inline-flex max-w-[200px] truncate text-xs font-bold text-slate-700 bg-slate-100 rounded-md px-2.5 py-1 border border-slate-200">
                               {worker.latestEvent.riskItem}
                            </span>
                            <span className="flex items-center gap-1 text-[10.5px] font-mono font-bold text-slate-400 ml-1">
                               <Clock className="w-3 h-3" /> {worker.latestEvent.timestamp}
                            </span>
                          </div>
                        ) : (
                          <span className="text-xs font-bold text-slate-400 bg-slate-50 px-2 py-1 rounded inline-block">위반 사항 없음</span>
                        )}
                     </td>

                     <td className="px-6 py-4 font-bold text-slate-700">
                        {worker.totalPenalty > 0 ? <span className="text-red-500">-{worker.totalPenalty}</span> : '0'}
                     </td>
                     <td className="px-6 py-4">
                        <span className={`inline-flex items-center justify-center px-3 py-1 rounded-full text-xs font-bold border whitespace-nowrap ${scoreColor} ${badgeBg}`}>
                          {worker.finalScore} 점
                        </span>
                     </td>
                     <td className="px-6 py-4 text-right">
                        <button className="inline-flex items-center justify-center gap-2 px-3 py-1.5 rounded-md bg-white border border-slate-200 text-slate-600 text-[11px] font-bold shadow-sm hover:bg-blue-50 focus:ring-2 focus:ring-blue-500 hover:text-blue-600 hover:border-blue-200 transition-all whitespace-nowrap">
                           <Search className="w-3.5 h-3.5" />
                           프로필 보기
                        </button>
                     </td>
                   </tr>
                 );
               })}
               {filteredScores.length === 0 && (
                 <tr>
                    <td colSpan={7} className="px-6 py-8 text-center text-slate-500 font-medium">
                       해당 조건의 작업자가 없습니다.
                    </td>
                 </tr>
               )}
             </tbody>
          </table>
       </div>
    </div>
  )
}
