export const workers = [
  { id: 'W001', name: '김현수', role: '지게차', baseScore: 100 },
  { id: 'W002', name: '박지훈', role: '크레인', baseScore: 100 },
  { id: 'W003', name: '이민수', role: '지게차', baseScore: 100 },
  { id: 'W004', name: '최동현', role: '크레인', baseScore: 100 },
  { id: 'W005', name: '정태영', role: '지게차', baseScore: 100 },
];

export const events = [
  { id: 'E001', workerId: 'W001', timestamp: '09:15', type: '지게차', riskItem: '속도 위반', penalty: 5 },
  { id: 'E002', workerId: 'W001', timestamp: '11:30', type: '지게차', riskItem: '보행자 근접', penalty: 12 },
  { id: 'E003', workerId: 'W002', timestamp: '10:45', type: '크레인', riskItem: '인양 반경 침범', penalty: 7 },
  { id: 'E004', workerId: 'W004', timestamp: '14:20', type: '크레인', riskItem: '하부 작업 시도', penalty: 30 },
  { id: 'E005', workerId: 'W005', timestamp: '16:05', type: '지게차', riskItem: '코너 미정지', penalty: 8 },
  { id: 'E006', workerId: 'W003', timestamp: '17:15', type: '지게차', riskItem: '불안정 하중', penalty: 10 },
  { id: 'E007', workerId: 'W004', timestamp: '17:30', type: '크레인', riskItem: '급격한 선회', penalty: 15 },
];

export const nodeScores = workers.map(worker => {
  // Sort events by timestamp to find the latest
  const workerEvents = events.filter(e => e.workerId === worker.id).sort((a,b) => a.timestamp.localeCompare(b.timestamp));
  const totalPenalty = workerEvents.reduce((sum, e) => sum + e.penalty, 0);
  const finalScore = Math.max(0, worker.baseScore - totalPenalty);
  const latestEvent = workerEvents.length > 0 ? workerEvents[workerEvents.length - 1] : null;
  return {
    ...worker,
    events: workerEvents,
    totalPenalty,
    finalScore,
    latestEvent
  };
}).sort((a, b) => a.finalScore - b.finalScore); // 위험군 상단 정렬 (오름차순)

// 현실적이고 뾰족한 다중 추이 데이터 (지게차 vs 크레인)
export const trendData = [
  { time: '09:00', forklift: 5, crane: 0 },
  { time: '09:30', forklift: 2, crane: 8 },
  { time: '10:00', forklift: 0, crane: 20 },
  { time: '10:30', forklift: 15, crane: 0 },
  { time: '11:00', forklift: 8, crane: 5 },
  { time: '11:30', forklift: 12, crane: 7 },
  { time: '12:00', forklift: 0, crane: 0 },
  { time: '13:00', forklift: 0, crane: 0 },
  { time: '13:30', forklift: 25, crane: 10 },
  { time: '14:00', forklift: 5, crane: 30 },
  { time: '14:30', forklift: 8, crane: 12 },
  { time: '15:00', forklift: 3, crane: 0 },
  { time: '15:30', forklift: 18, crane: 15 },
  { time: '16:00', forklift: 8, crane: 5 },
  { time: '16:30', forklift: 12, crane: 18 },
  { time: '17:00', forklift: 10, crane: 25 },
  { time: '17:30', forklift: 5, crane: 15 },
  { time: '18:00', forklift: 0, crane: 2 },
];

export const riskFactors = [
  { name: '크레인 하부 작업', value: 35 },
  { name: '지게차 과속', value: 25 },
  { name: '보행자 근접', value: 20 },
  { name: '안전모 미착용', value: 12 },
  { name: '급격한 선회', value: 8 },
];
export const RISK_COLORS = ['#8b5cf6', '#3b82f6', '#f59e0b', '#ef4444', '#94a3b8'];
