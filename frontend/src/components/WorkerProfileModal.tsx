import { User, ShieldAlert, PhoneCall } from 'lucide-react';

export default function WorkerProfileModal({ worker, onClose }: { worker: any, onClose: () => void }) {
  if (!worker) return null;
  const scoreColor = worker.finalScore >= 90 ? 'text-emerald-500' : worker.finalScore >= 70 ? 'text-amber-500' : 'text-red-500';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm p-4">
       <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md overflow-hidden flex flex-col border border-slate-200">
         <div className="flex justify-between items-center px-6 py-5 border-b border-slate-100 bg-slate-50">
            <div className="flex items-center gap-4">
               <div className="bg-blue-100 text-blue-600 p-2.5 rounded-full">
                 <User className="w-6 h-6" />
               </div>
               <div>
                  <h3 className="font-extrabold text-slate-800 text-lg tracking-tight">{worker.name}</h3>
                  <p className="text-sm text-slate-500 font-semibold">{worker.role}</p>
               </div>
            </div>
            <div className="text-right">
                <span className={`text-3xl font-extrabold tracking-tight block ${scoreColor}`}>{worker.finalScore}</span>
                <span className="text-[11px] uppercase tracking-widest text-slate-400 block font-bold mt-0.5">안전 점수</span>
            </div>
         </div>
         <div className="p-6 overflow-y-auto max-h-[50vh] bg-white">
            <h4 className="text-sm font-bold text-slate-800 mb-4 flex items-center">
               <ShieldAlert className="w-4 h-4 mr-2 text-slate-400" /> 오늘 발생한 감점 이벤트 내역
            </h4>
            {worker.events.length === 0 ? (
                <div className="text-center py-6 border border-dashed border-slate-200 rounded-xl bg-slate-50">
                  <p className="text-sm text-slate-500 font-medium">감점 이벤트가 없습니다.</p>
                </div>
            ) : (
                <ul className="space-y-3">
                   {worker.events.map((e: any) => (
                       <li key={e.id} className="flex justify-between items-center p-3.5 rounded-xl border border-slate-100 bg-[#f7f9fc]">
                          <div>
                             <span className="text-xs font-mono font-bold text-blue-500 mr-3">{e.timestamp}</span>
                             <span className="text-sm font-bold text-slate-700">{e.riskItem}</span>
                          </div>
                          <span className="text-sm font-extrabold text-red-500 bg-red-50 px-2 py-0.5 rounded border border-red-100">-{e.penalty}점</span>
                       </li>
                   ))}
                </ul>
            )}
         </div>
         <div className="px-6 py-4 border-t border-slate-100 bg-slate-50 flex justify-end gap-3">
            <button onClick={onClose} className="px-5 py-2.5 rounded-xl text-sm font-bold text-slate-600 bg-white border border-slate-200 hover:bg-slate-100 hover:text-slate-800 transition-colors shadow-sm">
              닫기
            </button>
            <button className="flex items-center px-5 py-2.5 rounded-xl text-sm font-bold text-white bg-blue-600 hover:bg-blue-700 transition-colors shadow-sm shadow-blue-600/20">
              <PhoneCall className="w-4 h-4 mr-2" /> 안전 교육 호출
            </button>
         </div>
       </div>
    </div>
  )
}
