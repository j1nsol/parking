import { useState, useEffect, useRef, useCallback } from "react";

const GLOBAL_STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #070a10; }
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 2px; }
  @keyframes fadeUp   { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:none} }
  @keyframes slideIn  { from{opacity:0;transform:translateX(18px)} to{opacity:1;transform:none} }
  @keyframes blink    { 0%,100%{opacity:1} 50%{opacity:0} }
  @keyframes pulse    { 0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(16,185,129,.4)} 50%{opacity:.7;box-shadow:0 0 0 7px rgba(16,185,129,0)} }
  @keyframes scanline { 0%{top:-10%} 100%{top:110%} }
  @keyframes shake    { 0%,100%{transform:translateX(0)} 25%{transform:translateX(-4px)} 75%{transform:translateX(4px)} }
  @keyframes spin     { to{transform:rotate(360deg)} }
  @keyframes dropzone { 0%,100%{border-color:rgba(56,189,248,.3)} 50%{border-color:rgba(56,189,248,.8)} }
`;

const C = {
  bg:"#070a10", surface:"#0d1018", card:"#111520",
  border:"rgba(255,255,255,0.07)",
  occ:"#f43f5e", vac:"#10b981", accent:"#38bdf8", warn:"#f59e0b", purple:"#a78bfa",
  text:"#e2e8f0", muted:"rgba(226,232,240,0.38)",
  mono:"'JetBrains Mono', monospace", sans:"'Syne', sans-serif",
};

const INITIAL_SLOTS = {
  S01:{status:"Occupied", coords:[40, 70, 155,165],row:"A",confidence:0.91},
  S02:{status:"Vacant",   coords:[170,70, 285,165],row:"A",confidence:0.87},
  S03:{status:"Occupied", coords:[300,70, 415,165],row:"A",confidence:0.95},
  S04:{status:"Vacant",   coords:[430,70, 545,165],row:"A",confidence:0.88},
  S05:{status:"Occupied", coords:[560,70, 675,165],row:"A",confidence:0.79},
  S06:{status:"Vacant",   coords:[40, 215,155,310],row:"B",confidence:0.93},
  S07:{status:"Occupied", coords:[170,215,285,310],row:"B",confidence:0.82},
  S08:{status:"Vacant",   coords:[300,215,415,310],row:"B",confidence:0.96},
  S09:{status:"Occupied", coords:[430,215,545,310],row:"B",confidence:0.74},
  S10:{status:"Vacant",   coords:[560,215,675,310],row:"B",confidence:0.89},
  S11:{status:"Occupied", coords:[40, 360,155,455],row:"C",confidence:0.91},
  S12:{status:"Vacant",   coords:[170,360,285,455],row:"C",confidence:0.85},
  S13:{status:"Occupied", coords:[300,360,415,455],row:"C",confidence:0.97},
  S14:{status:"Vacant",   coords:[430,360,545,455],row:"C",confidence:0.88},
  S15:{status:"Occupied", coords:[560,360,675,455],row:"C",confidence:0.76},
};

const pct = (a,b) => b ? Math.round((a/b)*100) : 0;
const rnd = (min,max) => +(Math.random()*(max-min)+min).toFixed(2);
const fmtTs = () => new Date().toLocaleTimeString("en-PH",{hour12:false});

// ── Primitives ────────────────────────────────────────────────────────────────
function LiveDot({color=C.vac,size=8}){
  return <span style={{display:"inline-block",width:size,height:size,borderRadius:"50%",background:color,flexShrink:0,animation:"pulse 2s infinite"}}/>;
}
function Badge({label,color}){
  return <span style={{padding:"2px 8px",borderRadius:4,fontSize:10,fontWeight:700,fontFamily:C.mono,letterSpacing:"0.06em",textTransform:"uppercase",background:`${color}22`,color,border:`1px solid ${color}44`}}>{label}</span>;
}
function Card({children,style={}}){
  return <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:16,...style}}>{children}</div>;
}
function StatPill({label,value,color,sub}){
  return(
    <div style={{flex:1,minWidth:110,padding:"18px 20px",background:C.card,border:`1px solid ${C.border}`,borderRadius:14,position:"relative",overflow:"hidden"}}>
      <div style={{position:"absolute",inset:0,opacity:.07,background:`radial-gradient(circle at 80% 20%,${color},transparent 70%)`}}/>
      <div style={{fontSize:30,fontWeight:800,fontFamily:C.sans,color,lineHeight:1}}>{value}</div>
      <div style={{fontSize:11,fontFamily:C.mono,color:C.muted,marginTop:5,textTransform:"uppercase",letterSpacing:"0.08em"}}>{label}</div>
      {sub&&<div style={{fontSize:10,color,marginTop:2,fontFamily:C.mono}}>{sub}</div>}
    </div>
  );
}

// ── Parking Map SVG ───────────────────────────────────────────────────────────
function ParkingMap({slots,selectedSlot,onSelect,adminMode,onRemove}){
  return(
    <div style={{width:"100%",overflowX:"auto"}}>
      <svg viewBox="0 0 720 530" style={{width:"100%",minWidth:320,display:"block"}}>
        <rect width="720" height="530" rx="14" fill="#080c15"/>
        <defs>
          <pattern id="g" width="30" height="30" patternUnits="userSpaceOnUse">
            <path d="M30 0L0 0 0 30" fill="none" stroke="rgba(255,255,255,.03)" strokeWidth=".5"/>
          </pattern>
          <filter id="glow"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>
        <rect width="720" height="530" fill="url(#g)" rx="14"/>
        {[190,335].map((y,i)=>(
          <g key={i}>
            <rect x="20" y={y} width="680" height="30" fill="rgba(255,255,255,.025)" rx="2"/>
            <text x="360" y={y+20} textAnchor="middle" fill="rgba(255,255,255,.1)" fontSize="9" fontFamily={C.mono} letterSpacing="4">▶  DRIVE LANE  ▶</text>
          </g>
        ))}
        {["A","B","C"].map((r,i)=>(
          <text key={r} x="14" y={[130,275,420][i]} textAnchor="middle" fill="rgba(255,255,255,.18)" fontSize="11" fontWeight="700" fontFamily={C.mono}>{r}</text>
        ))}
        <rect x="300" y="492" width="120" height="24" rx="12" fill="rgba(251,191,36,.1)" stroke="#fbbf24" strokeWidth="1"/>
        <text x="360" y="508" textAnchor="middle" fill="#fbbf24" fontSize="9" fontFamily={C.mono} letterSpacing="3">ENTRANCE</text>
        <line x1="360" y1="491" x2="360" y2="483" stroke="#fbbf24" strokeWidth="1" strokeDasharray="3,2"/>
        {Object.entries(slots).map(([id,slot])=>{
          const [x1,y1,x2,y2]=slot.coords, w=x2-x1, h=y2-y1;
          const occ=slot.status==="Occupied", sel=selectedSlot===id;
          const stroke=occ?C.occ:C.vac;
          return(
            <g key={id} onClick={()=>onSelect(sel?null:id)} style={{cursor:"pointer"}}>
              <rect x={x1} y={y1} width={w} height={h} rx="6"
                fill={sel?(occ?`${C.occ}50`:`${C.vac}45`):(occ?`${C.occ}28`:`${C.vac}20`)}
                stroke={stroke} strokeWidth={sel?2.5:1.5} style={{transition:"all .3s"}}
                filter={sel?"url(#glow)":undefined}/>
              {occ&&<text x={x1+w/2} y={y1+h/2-4} textAnchor="middle" fontSize="18" style={{userSelect:"none"}}>🚗</text>}
              {!occ&&<circle cx={x1+w/2} cy={y1+h/2-6} r="7" fill={C.vac} opacity=".35"/>}
              <text x={x1+w/2} y={y1+h-9} textAnchor="middle" fill={occ?"#fda4af":"#6ee7b7"} fontSize="10" fontFamily={C.mono} fontWeight="700">{id}</text>
              <rect x={x1+4} y={y2-5} width={w-8} height={2} rx="1" fill="rgba(255,255,255,.08)"/>
              <rect x={x1+4} y={y2-5} width={(w-8)*(slot.confidence||.8)} height={2} rx="1" fill={occ?C.occ:C.vac} opacity=".65"/>
              {adminMode&&sel&&(
                <g onClick={e=>{e.stopPropagation();onRemove(id);}}>
                  <rect x={x2-22} y={y1+2} width={20} height={20} rx="4" fill="#ef444488" stroke="#ef4444" strokeWidth="1"/>
                  <text x={x2-12} y={y1+15} textAnchor="middle" fill="#fff" fontSize="12" fontWeight="700">✕</text>
                </g>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ── AI Terminal ───────────────────────────────────────────────────────────────
function AITerminal({logs}){
  const ref=useRef(null);
  useEffect(()=>{ref.current?.scrollIntoView({behavior:"smooth"});},[logs]);
  return(
    <div style={{background:"#04060d",border:`1px solid rgba(56,189,248,.2)`,borderRadius:12,padding:"14px 16px",height:260,overflowY:"auto",fontFamily:C.mono,fontSize:11,position:"relative"}}>
      <div style={{position:"absolute",left:0,right:0,height:"2px",background:"linear-gradient(90deg,transparent,rgba(56,189,248,.15),transparent)",animation:"scanline 4s linear infinite",pointerEvents:"none"}}/>
      <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:10,paddingBottom:8,borderBottom:`1px solid rgba(255,255,255,.05)`}}>
        <LiveDot color={C.accent}/>
        <span style={{color:C.accent,fontWeight:700,letterSpacing:"0.1em",fontSize:10}}>AI PROCESSING FEED</span>
        <span style={{marginLeft:"auto",color:C.muted,fontSize:10}}>{logs.length} events</span>
      </div>
      {logs.map((l,i)=>(
        <div key={l.id} style={{padding:"2px 0",color:l.type==="error"?C.occ:l.type==="sync"?C.vac:l.type==="sys"?C.warn:l.type==="img"?C.purple:"rgba(148,163,184,.85)",animation:i===logs.length-1?"fadeUp .25s ease":"none",display:"flex",gap:10}}>
          <span style={{color:"rgba(255,255,255,.18)",flexShrink:0}}>{l.time}</span>
          <span>{l.msg}</span>
        </div>
      ))}
      <div ref={ref}/>
    </div>
  );
}

// ── Confirm Dialog ────────────────────────────────────────────────────────────
function ConfirmDialog({slotId,onConfirm,onCancel}){
  return(
    <div style={{position:"fixed",inset:0,background:"rgba(0,0,0,.75)",display:"flex",alignItems:"center",justifyContent:"center",zIndex:999,backdropFilter:"blur(6px)"}}>
      <div style={{background:C.card,border:`1px solid ${C.occ}44`,borderRadius:20,padding:32,maxWidth:360,width:"90%",animation:"fadeUp .2s ease",boxShadow:`0 0 60px ${C.occ}22`}}>
        <div style={{fontSize:32,marginBottom:12,textAlign:"center"}}>⚠️</div>
        <div style={{fontFamily:C.sans,fontWeight:700,fontSize:18,textAlign:"center",marginBottom:8}}>Remove Slot {slotId}?</div>
        <div style={{color:C.muted,fontSize:13,textAlign:"center",marginBottom:24,lineHeight:1.6}}>
          This removes <strong style={{color:C.text}}>{slotId}</strong> from the map and stops monitoring it.
        </div>
        <div style={{display:"flex",gap:12}}>
          <button onClick={onCancel} style={{flex:1,padding:"12px",borderRadius:10,fontFamily:C.sans,fontWeight:600,fontSize:14,cursor:"pointer",background:"rgba(255,255,255,.06)",border:`1px solid ${C.border}`,color:C.text}}>Cancel</button>
          <button onClick={onConfirm} style={{flex:1,padding:"12px",borderRadius:10,fontFamily:C.sans,fontWeight:700,fontSize:14,cursor:"pointer",background:C.occ,border:"none",color:"#fff",animation:"shake .4s ease"}}>Remove</button>
        </div>
      </div>
    </div>
  );
}

// ── Slot Detail ───────────────────────────────────────────────────────────────
function SlotDetail({slotId,slot,onClose,onRemove,adminMode}){
  if(!slot) return null;
  const occ=slot.status==="Occupied";
  return(
    <div style={{background:"rgba(255,255,255,.03)",border:`1px solid ${occ?C.occ+"44":C.vac+"44"}`,borderRadius:14,padding:18,marginTop:14,animation:"slideIn .25s ease"}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:14}}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <span style={{fontFamily:C.sans,fontWeight:800,fontSize:20,color:occ?C.occ:C.vac}}>Slot {slotId}</span>
          <Badge label={slot.status} color={occ?C.occ:C.vac}/>
        </div>
        <button onClick={onClose} style={{background:"none",border:"none",color:C.muted,cursor:"pointer",fontSize:18}}>✕</button>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,marginBottom:adminMode?14:0}}>
        {[["Row",`Row ${slot.row}`],["Confidence",`${Math.round((slot.confidence||.8)*100)}%`],["Sensor","Camera (YOLO)"],["Coords",`[${slot.coords.join(",")}]`]].map(([k,v])=>(
          <div key={k} style={{background:"rgba(0,0,0,.3)",borderRadius:8,padding:"8px 12px"}}>
            <div style={{fontSize:9,color:C.muted,textTransform:"uppercase",letterSpacing:"0.1em",fontFamily:C.mono}}>{k}</div>
            <div style={{fontSize:12,fontWeight:600,marginTop:3,fontFamily:C.mono,color:C.text}}>{v}</div>
          </div>
        ))}
      </div>
      {adminMode&&(
        <button onClick={()=>onRemove(slotId)} style={{width:"100%",padding:"10px",borderRadius:10,background:`${C.occ}18`,border:`1px solid ${C.occ}55`,color:C.occ,fontFamily:C.sans,fontWeight:700,fontSize:13,cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center",gap:8}}>
          <span>✕</span> Remove This Slot
        </button>
      )}
    </div>
  );
}

// ── IMAGE TEST PANEL ──────────────────────────────────────────────────────────
function ImageTestPanel({onAnalysisComplete,addLog}){
  const [dragOver,setDragOver]   = useState(false);
  const [image,setImage]         = useState(null);
  const [analyzing,setAnalyzing] = useState(false);
  const [result,setResult]       = useState(null);
  const [error,setError]         = useState(null);
  const inputRef                 = useRef(null);

  const handleFile = (file) => {
    if(!file||!file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = e => { setImage(e.target.result); setResult(null); setError(null); };
    reader.readAsDataURL(file);
  };

  const analyzeImage = async () => {
    if(!image) return;
    setAnalyzing(true); setError(null);
    addLog("[IMG]  Image uploaded — sending to AI for analysis...","img");
    try {
      const base64    = image.split(",")[1];
      const mediaType = image.split(";")[0].split(":")[1];
      const response  = await fetch("https://api.anthropic.com/v1/messages",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({
          model:"claude-sonnet-4-20250514",
          max_tokens:1000,
          messages:[{
            role:"user",
            content:[
              {type:"image",source:{type:"base64",media_type:mediaType,data:base64}},
              {type:"text",text:`You are a parking lot AI detection system. Analyze this parking lot image carefully.

Respond ONLY with a valid JSON object in this exact format (no extra text, no markdown):
{
  "total_slots_detected": <number>,
  "occupied_slots": <number>,
  "vacant_slots": <number>,
  "occupancy_percent": <number 0-100>,
  "confidence": <number 0.0-1.0>,
  "lighting": "<Good|Fair|Poor>",
  "notes": "<one concise sentence about what you see>",
  "slots": [
    { "id": "S01", "status": "Occupied", "row": "A", "confidence": 0.92 },
    { "id": "S02", "status": "Vacant",   "row": "A", "confidence": 0.88 }
  ]
}

Rules:
- Assign row A to top third of image, B to middle, C to bottom
- If not a parking lot, set total_slots_detected to 0
- Be realistic with confidence scores
- Number slots S01, S02... in order left-to-right, top-to-bottom`}
            ]
          }]
        })
      });
      const data   = await response.json();
      const raw    = data.content?.map(c=>c.text||"").join("")||"";
      const clean  = raw.replace(/```json|```/g,"").trim();
      const parsed = JSON.parse(clean);
      setResult(parsed);
      addLog(`[IMG]  AI analysis complete — ${parsed.total_slots_detected} slots detected`,"img");
      addLog(`[IMG]  Occupied: ${parsed.occupied_slots} | Vacant: ${parsed.vacant_slots} | Conf: ${Math.round(parsed.confidence*100)}%`,"img");
      addLog(`[IMG]  Lighting: ${parsed.lighting} | ${parsed.notes}`,"img");
      if(onAnalysisComplete && parsed.total_slots_detected>0) onAnalysisComplete(parsed);
    } catch(err){
      setError("AI analysis failed — could not parse response. Try a clearer parking lot image.");
      addLog("[IMG]  ERROR: AI analysis failed","error");
    } finally { setAnalyzing(false); }
  };

  return(
    <div style={{display:"flex",flexDirection:"column",gap:16}}>
      {/* Header */}
      <div style={{padding:"14px 18px",background:`linear-gradient(135deg,rgba(167,139,250,.1),rgba(56,189,248,.06))`,border:`1px solid rgba(167,139,250,.25)`,borderRadius:14,display:"flex",alignItems:"center",gap:12}}>
        <span style={{fontSize:24}}>🖼️</span>
        <div>
          <div style={{fontFamily:C.sans,fontWeight:800,fontSize:15,color:C.purple}}>Image Testing Mode</div>
          <div style={{fontSize:11,fontFamily:C.mono,color:C.muted}}>Upload a parking lot photo — Claude AI will analyze occupancy</div>
        </div>
        <Badge label="TESTING" color={C.purple}/>
      </div>

      {/* When to use this */}
      <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
        {["📡 No live camera feed","🧪 Testing accuracy","🔧 Development mode","✅ Verify AI works"].map(t=>(
          <div key={t} style={{padding:"5px 12px",borderRadius:8,background:"rgba(255,255,255,.04)",border:`1px solid ${C.border}`,fontFamily:C.mono,fontSize:10,color:C.muted}}>{t}</div>
        ))}
      </div>

      {/* Drop zone */}
      <div
        onDragOver={e=>{e.preventDefault();setDragOver(true)}}
        onDragLeave={()=>setDragOver(false)}
        onDrop={e=>{e.preventDefault();setDragOver(false);handleFile(e.dataTransfer.files[0]);}}
        onClick={()=>inputRef.current?.click()}
        style={{border:`2px dashed ${dragOver?C.accent:"rgba(56,189,248,.3)"}`,borderRadius:16,padding:"28px 20px",textAlign:"center",cursor:"pointer",background:dragOver?"rgba(56,189,248,.06)":"rgba(255,255,255,.02)",transition:"all .2s",animation:!image&&!dragOver?"dropzone 2.5s ease infinite":"none"}}
      >
        <input ref={inputRef} type="file" accept="image/*" style={{display:"none"}} onChange={e=>handleFile(e.target.files[0])}/>
        {image ? (
          <div>
            <img src={image} alt="uploaded" style={{maxHeight:260,maxWidth:"100%",borderRadius:10,border:`1px solid ${C.border}`,objectFit:"contain"}}/>
            <div style={{marginTop:10,fontSize:11,fontFamily:C.mono,color:C.muted}}>✓ Image ready — click Analyze or drop a new image to replace</div>
          </div>
        ):(
          <div>
            <div style={{fontSize:44,marginBottom:12}}>📷</div>
            <div style={{fontFamily:C.sans,fontWeight:700,fontSize:16,marginBottom:6}}>Drop a parking lot image here</div>
            <div style={{fontSize:11,fontFamily:C.mono,color:C.muted,marginBottom:14}}>or click to browse · JPG, PNG, WEBP</div>
            <div style={{display:"inline-flex",gap:16,padding:"8px 18px",borderRadius:10,background:"rgba(255,255,255,.04)",border:`1px solid ${C.border}`,fontSize:11,fontFamily:C.mono,color:C.muted}}>
              <span>📸 Screenshot</span><span>🛰️ Overhead</span><span>🖥️ Test image</span>
            </div>
          </div>
        )}
      </div>

      {/* Analyze button */}
      {image&&(
        <button onClick={analyzeImage} disabled={analyzing} style={{padding:"14px",borderRadius:12,border:"none",cursor:analyzing?"not-allowed":"pointer",fontFamily:C.sans,fontWeight:800,fontSize:15,background:analyzing?"rgba(167,139,250,.2)":`linear-gradient(135deg,${C.purple},${C.accent})`,color:"#fff",display:"flex",alignItems:"center",justifyContent:"center",gap:10,opacity:analyzing?.7:1,transition:"all .2s"}}>
          {analyzing
            ?<><span style={{width:18,height:18,border:"2px solid rgba(255,255,255,.3)",borderTopColor:"#fff",borderRadius:"50%",animation:"spin .8s linear infinite",display:"inline-block"}}/>Analyzing with AI...</>
            :<>🔍 Analyze Parking Lot Image</>}
        </button>
      )}

      {error&&<div style={{padding:"12px 16px",borderRadius:10,background:`${C.occ}15`,border:`1px solid ${C.occ}44`,fontFamily:C.mono,fontSize:12,color:C.occ}}>⚠️ {error}</div>}

      {/* Results */}
      {result&&(
        <div style={{animation:"fadeUp .3s ease",display:"flex",flexDirection:"column",gap:12}}>
          <div style={{padding:"14px 18px",background:`linear-gradient(135deg,rgba(16,185,129,.08),rgba(56,189,248,.05))`,border:`1px solid rgba(16,185,129,.25)`,borderRadius:14}}>
            <div style={{fontFamily:C.sans,fontWeight:700,fontSize:14,marginBottom:12,color:C.vac}}>✅ AI Analysis Results</div>
            <div style={{display:"flex",gap:10,flexWrap:"wrap",marginBottom:12}}>
              {[["Slots Found",result.total_slots_detected,C.accent],["Occupied",result.occupied_slots,C.occ],["Vacant",result.vacant_slots,C.vac],["Confidence",`${Math.round(result.confidence*100)}%`,C.purple]].map(([l,v,c])=>(
                <div key={l} style={{flex:1,minWidth:80,padding:"12px 14px",background:"rgba(0,0,0,.3)",borderRadius:10,textAlign:"center"}}>
                  <div style={{fontSize:24,fontWeight:800,fontFamily:C.sans,color:c,lineHeight:1}}>{v}</div>
                  <div style={{fontSize:9,fontFamily:C.mono,color:C.muted,marginTop:4,textTransform:"uppercase",letterSpacing:"0.08em"}}>{l}</div>
                </div>
              ))}
            </div>
            <div style={{marginBottom:10}}>
              <div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}>
                <span style={{fontFamily:C.mono,fontSize:11,color:C.muted}}>Occupancy</span>
                <span style={{fontFamily:C.mono,fontSize:11,fontWeight:700,color:result.occupancy_percent>75?C.occ:C.vac}}>{result.occupancy_percent}%</span>
              </div>
              <div style={{height:8,background:"rgba(255,255,255,.07)",borderRadius:4,overflow:"hidden"}}>
                <div style={{height:"100%",width:`${result.occupancy_percent}%`,background:result.occupancy_percent>75?`linear-gradient(90deg,${C.occ},#dc2626)`:`linear-gradient(90deg,${C.vac},#059669)`,borderRadius:4}}/>
              </div>
            </div>
            <div style={{display:"flex",gap:8,marginBottom:10,flexWrap:"wrap"}}>
              <Badge label={`Lighting: ${result.lighting}`} color={result.lighting==="Good"?C.vac:result.lighting==="Fair"?C.warn:C.occ}/>
              <Badge label={`${result.total_slots_detected} slots detected`} color={C.accent}/>
            </div>
            <div style={{padding:"10px 14px",borderRadius:8,background:"rgba(255,255,255,.04)",fontFamily:C.mono,fontSize:11,color:"rgba(226,232,240,.7)",lineHeight:1.6}}>💬 {result.notes}</div>
          </div>

          {result.slots?.length>0&&(
            <div>
              <div style={{fontFamily:C.sans,fontWeight:700,fontSize:13,marginBottom:10,color:C.muted}}>Detected Slots ({result.slots.length})</div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(88px,1fr))",gap:8,marginBottom:12}}>
                {result.slots.map(s=>{
                  const occ=s.status==="Occupied";
                  return(
                    <div key={s.id} style={{padding:"10px 6px",borderRadius:10,textAlign:"center",background:"rgba(255,255,255,.03)",border:`1px solid ${occ?C.occ+"33":C.vac+"25"}`}}>
                      <div style={{fontSize:15,marginBottom:4}}>{occ?"🚗":"🟢"}</div>
                      <div style={{fontSize:10,fontFamily:C.mono,fontWeight:700,color:occ?"#fda4af":"#6ee7b7"}}>{s.id}</div>
                      <div style={{fontSize:8,color:C.muted,marginTop:2,fontFamily:C.mono}}>Row {s.row}</div>
                      <div style={{fontSize:8,color:C.muted,marginTop:1,fontFamily:C.mono}}>{Math.round(s.confidence*100)}%</div>
                    </div>
                  );
                })}
              </div>
              <button onClick={()=>onAnalysisComplete(result)} style={{width:"100%",padding:"12px",borderRadius:12,border:`1px solid ${C.purple}55`,background:`${C.purple}18`,color:C.purple,fontFamily:C.sans,fontWeight:700,fontSize:13,cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center",gap:8}}>
                🗺️ Apply These Results to the Parking Map
              </button>
            </div>
          )}

          {result.total_slots_detected===0&&(
            <div style={{padding:"14px 18px",borderRadius:12,background:`${C.warn}12`,border:`1px solid ${C.warn}44`,fontFamily:C.mono,fontSize:12,color:C.warn,textAlign:"center"}}>
              ⚠️ No parking slots detected. Try an overhead or angled photo showing parking slot markings.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Admin Panel ───────────────────────────────────────────────────────────────
function AdminPanel({slots,logs,onRemove,removedSlots,addLog,onImageAnalysis}){
  const [selected,setSelected]     = useState(null);
  const [confirm,setConfirm]       = useState(null);
  const [section,setSection]       = useState("map");
  const total    = Object.keys(slots).length;
  const occupied = Object.values(slots).filter(s=>s.status==="Occupied").length;
  const vacant   = total - occupied;

  return(
    <div style={{display:"flex",flexDirection:"column",gap:18}}>
      <div style={{padding:"14px 18px",background:`linear-gradient(135deg,rgba(244,63,94,.08),rgba(251,191,36,.05))`,border:`1px solid rgba(244,63,94,.2)`,borderRadius:14,display:"flex",alignItems:"center",gap:12}}>
        <span style={{fontSize:22}}>🛡️</span>
        <div>
          <div style={{fontFamily:C.sans,fontWeight:800,fontSize:15,color:"#fda4af"}}>Admin Panel</div>
          <div style={{fontSize:11,fontFamily:C.mono,color:C.muted}}>Full system access · CIT-U Parking</div>
        </div>
        <div style={{marginLeft:"auto",display:"flex",gap:8}}>
          <Badge label="LIVE" color={C.vac}/>
          <Badge label="ADMIN" color={C.occ}/>
        </div>
      </div>

      <div style={{display:"flex",gap:10,flexWrap:"wrap"}}>
        <StatPill label="Active Slots" value={total}    color={C.accent}/>
        <StatPill label="Occupied"     value={occupied} color={C.occ}   sub={`${pct(occupied,total)}% full`}/>
        <StatPill label="Available"    value={vacant}   color={C.vac}/>
        <StatPill label="Removed"      value={removedSlots.length} color={C.warn}/>
      </div>

      {/* Tabs */}
      <div style={{display:"flex",gap:2,background:C.surface,borderRadius:12,padding:4,border:`1px solid ${C.border}`}}>
        {[{id:"map",label:"🗺️ Live Map"},{id:"image",label:"🖼️ Image Test"},{id:"logs",label:"📡 AI Feed"}].map(s=>(
          <button key={s.id} onClick={()=>setSection(s.id)} style={{flex:1,padding:"9px 8px",borderRadius:9,border:"none",cursor:"pointer",fontFamily:C.sans,fontWeight:700,fontSize:12,background:section===s.id?"rgba(255,255,255,.08)":"transparent",color:section===s.id?C.text:C.muted,transition:"all .2s"}}>{s.label}</button>
        ))}
      </div>

      {section==="map"&&(
        <Card style={{padding:20}}>
          <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:14}}>
            <span style={{fontFamily:C.sans,fontWeight:700,fontSize:15}}>Live Parking Map</span>
            <div style={{display:"flex",alignItems:"center",gap:6}}>
              <LiveDot/>
              <span style={{fontFamily:C.mono,fontSize:10,color:C.muted}}>Click slot → ✕ to remove</span>
            </div>
          </div>
          <ParkingMap slots={slots} selectedSlot={selected} onSelect={setSelected} adminMode={true} onRemove={id=>setConfirm(id)}/>
          <SlotDetail slotId={selected} slot={selected?slots[selected]:null} onClose={()=>setSelected(null)} onRemove={id=>setConfirm(id)} adminMode={true}/>
          <div style={{marginTop:20,paddingTop:20,borderTop:`1px solid ${C.border}`}}>
            <div style={{fontFamily:C.sans,fontWeight:700,fontSize:13,marginBottom:12,color:C.muted}}>Row Breakdown</div>
            {["A","B","C"].map(row=>{
              const rs=Object.values(slots).filter(s=>s.row===row);
              const o=rs.filter(s=>s.status==="Occupied").length, t=rs.length, p=pct(o,t);
              return(
                <div key={row} style={{marginBottom:12}}>
                  <div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}>
                    <span style={{fontFamily:C.mono,fontSize:12}}>Row {row}</span>
                    <span style={{fontFamily:C.mono,fontSize:12,color:p>75?C.occ:p>50?C.warn:C.vac,fontWeight:700}}>{o}/{t} ({p}%)</span>
                  </div>
                  <div style={{height:7,background:"rgba(255,255,255,.07)",borderRadius:3,overflow:"hidden"}}>
                    <div style={{height:"100%",width:`${p}%`,background:p>75?`linear-gradient(90deg,${C.occ},#dc2626)`:p>50?`linear-gradient(90deg,${C.warn},#d97706)`:`linear-gradient(90deg,${C.vac},#059669)`,borderRadius:3,transition:"width .8s ease"}}/>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {section==="image"&&(
        <Card style={{padding:20}}>
          <ImageTestPanel onAnalysisComplete={onImageAnalysis} addLog={addLog}/>
        </Card>
      )}

      {section==="logs"&&(
        <Card style={{padding:20}}>
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:14}}>
            <span style={{fontFamily:C.sans,fontWeight:700,fontSize:15}}>AI Processing Feed</span>
            <span style={{fontFamily:C.mono,fontSize:10,color:C.accent,animation:"blink 1.2s infinite",marginLeft:4}}>●</span>
            <span style={{fontFamily:C.mono,fontSize:10,color:C.muted}}>YOLO v5n · Raspberry Pi 5</span>
          </div>
          <AITerminal logs={logs}/>
        </Card>
      )}

      {removedSlots.length>0&&(
        <Card style={{padding:20}}>
          <div style={{fontFamily:C.sans,fontWeight:700,fontSize:14,marginBottom:12,color:C.warn}}>⚠️ Removed Slots ({removedSlots.length})</div>
          <div style={{display:"flex",flexWrap:"wrap",gap:8}}>
            {removedSlots.map(({id,time})=>(
              <div key={id+time} style={{padding:"6px 12px",borderRadius:8,background:"rgba(245,158,11,.1)",border:"1px solid rgba(245,158,11,.3)",fontFamily:C.mono,fontSize:11}}>
                <span style={{color:C.warn,fontWeight:700}}>{id}</span>
                <span style={{color:C.muted,marginLeft:8}}>{time}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {confirm&&(
        <ConfirmDialog slotId={confirm}
          onConfirm={()=>{onRemove(confirm);setConfirm(null);setSelected(null);}}
          onCancel={()=>setConfirm(null)}/>
      )}
    </div>
  );
}

// ── Driver View ───────────────────────────────────────────────────────────────
function UserView({slots}){
  const [selected,setSelected] = useState(null);
  const [filter,setFilter]     = useState("All");
  const total    = Object.keys(slots).length;
  const occupied = Object.values(slots).filter(s=>s.status==="Occupied").length;
  const vacant   = total-occupied, p=pct(occupied,total);
  const filtered = Object.fromEntries(Object.entries(slots).filter(([,s])=>filter==="All"||s.status===filter));
  return(
    <div style={{display:"flex",flexDirection:"column",gap:18}}>
      <div style={{display:"flex",gap:10,flexWrap:"wrap"}}>
        <StatPill label="Total Slots" value={total}    color={C.accent}/>
        <StatPill label="Occupied"    value={occupied} color={C.occ}  sub={`${p}% full`}/>
        <StatPill label="Available"   value={vacant}   color={C.vac}  sub="Free now"/>
      </div>
      <Card style={{padding:18}}>
        <div style={{display:"flex",justifyContent:"space-between",marginBottom:8}}>
          <span style={{fontFamily:C.sans,fontWeight:600,fontSize:13,color:C.muted}}>Lot Capacity</span>
          <span style={{fontFamily:C.mono,fontWeight:700,fontSize:13,color:p>80?C.occ:p>60?C.warn:C.vac}}>{p}% occupied</span>
        </div>
        <div style={{height:10,background:"rgba(255,255,255,.07)",borderRadius:5,overflow:"hidden"}}>
          <div style={{height:"100%",width:`${p}%`,background:p>80?`linear-gradient(90deg,${C.occ},#dc2626)`:p>60?`linear-gradient(90deg,${C.warn},#d97706)`:`linear-gradient(90deg,${C.vac},#059669)`,borderRadius:5,transition:"width .8s ease"}}/>
        </div>
        {vacant===0&&<div style={{marginTop:10,padding:"8px 14px",borderRadius:8,background:`${C.occ}15`,border:`1px solid ${C.occ}44`,fontFamily:C.mono,fontSize:11,color:C.occ,textAlign:"center"}}>🚫 Parking lot is full — please check back shortly</div>}
      </Card>
      <Card style={{padding:20}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:14}}>
          <span style={{fontFamily:C.sans,fontWeight:700,fontSize:15}}>Find a Spot</span>
          <div style={{display:"flex",gap:8}}>
            {["All","Vacant","Occupied"].map(f=>(
              <button key={f} onClick={()=>setFilter(f)} style={{padding:"4px 12px",borderRadius:20,fontSize:11,fontWeight:700,fontFamily:C.mono,cursor:"pointer",border:filter===f?"none":`1px solid ${C.border}`,background:filter===f?f==="Vacant"?"#10b98133":f==="Occupied"?`${C.occ}33`:"#38bdf833":"transparent",color:filter===f?f==="Vacant"?C.vac:f==="Occupied"?C.occ:C.accent:C.muted}}>{f}</button>
            ))}
          </div>
        </div>
        <ParkingMap slots={filtered} selectedSlot={selected} onSelect={setSelected} adminMode={false} onRemove={()=>{}}/>
        <SlotDetail slotId={selected} slot={selected?slots[selected]:null} onClose={()=>setSelected(null)} onRemove={()=>{}} adminMode={false}/>
        <div style={{display:"flex",gap:20,marginTop:14,justifyContent:"center"}}>
          {[[C.vac,"Available"],[C.occ,"Occupied"]].map(([c,l])=>(
            <div key={l} style={{display:"flex",alignItems:"center",gap:6,fontSize:11,fontFamily:C.mono,color:C.muted}}>
              <span style={{width:10,height:10,borderRadius:2,background:c,display:"inline-block"}}/>{l}
            </div>
          ))}
        </div>
      </Card>
      <Card style={{padding:20}}>
        <div style={{fontFamily:C.sans,fontWeight:700,fontSize:15,marginBottom:14}}>All Slots <span style={{fontSize:12,fontWeight:400,color:C.muted,marginLeft:8}}>({Object.keys(filtered).length})</span></div>
        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(80px,1fr))",gap:8}}>
          {Object.entries(filtered).map(([id,slot])=>{
            const occ=slot.status==="Occupied";
            return(
              <div key={id} onClick={()=>setSelected(id===selected?null:id)} style={{padding:"10px 6px",borderRadius:10,textAlign:"center",cursor:"pointer",background:selected===id?(occ?`${C.occ}30`:`${C.vac}30`):"rgba(255,255,255,.03)",border:`1px solid ${occ?C.occ+"33":C.vac+"25"}`,transition:"all .2s"}}>
                <div style={{fontSize:15,marginBottom:4}}>{occ?"🚗":"🟢"}</div>
                <div style={{fontSize:10,fontFamily:C.mono,fontWeight:700,color:occ?"#fda4af":"#6ee7b7"}}>{id}</div>
                <div style={{fontSize:8,color:C.muted,marginTop:2,fontFamily:C.mono}}>Row {slot.row}</div>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}

// ── Root ──────────────────────────────────────────────────────────────────────
export default function App(){
  const [slots,setSlots]           = useState(INITIAL_SLOTS);
  const [tab,setTab]               = useState("user");
  const [logs,setLogs]             = useState([]);
  const [removed,setRemoved]       = useState([]);
  const logId                      = useRef(0);

  const addLog = useCallback((msg,type="info")=>{
    setLogs(p=>[...p.slice(-150),{id:logId.current++,msg,type,time:fmtTs()}]);
  },[]);

  useEffect(()=>{
    const iv=setInterval(()=>{
      setSlots(p=>{
        const keys=Object.keys(p); if(!keys.length) return p;
        const key=keys[Math.floor(Math.random()*keys.length)];
        const next=p[key].status==="Occupied"?"Vacant":"Occupied";
        const conf=rnd(.72,.97);
        addLog(`[YOLO] Slot ${key} → ${next==="Occupied"?"vehicle detected":"slot clear"} (conf: ${conf})`);
        return{...p,[key]:{...p[key],status:next,confidence:conf}};
      });
      const msgs=[
        ()=>`[FRAME] Processing frame #${String(Math.floor(Math.random()*9999)).padStart(4,"0")}`,
        ()=>`[SYNC]  Firebase push → OK (${Math.floor(Math.random()*12+2)}ms)`,
        ()=>`[YOLO] Inference time: ${(Math.random()*40+80).toFixed(1)}ms`,
        ()=>`[SYS]  CPU: ${Math.floor(Math.random()*20+45)}% | Temp: ${Math.floor(Math.random()*8+52)}°C`,
      ];
      const m=msgs[Math.floor(Math.random()*msgs.length)]();
      addLog(m,m.includes("SYNC")?"sync":m.includes("SYS")?"sys":"info");
    },2500);
    return()=>clearInterval(iv);
  },[addLog]);

  useEffect(()=>{
    ["[SYS]  Raspberry Pi 5 booting...","[SYS]  Camera initialized (1080p USB, IR)","[YOLO] Loading yolov5n.pt...","[YOLO] Model ready — 15 slots loaded","[MAP]  Auto-mapping complete","[FB]   Firebase connected ✓","[SYS]  Image upload fallback enabled"].forEach((m,i)=>
      setTimeout(()=>addLog(m,m.includes("FB")?"sync":"sys"),i*280));
  },[]);

  const handleRemove = useCallback((id)=>{
    setSlots(p=>{const n={...p};delete n[id];return n;});
    setRemoved(p=>[...p,{id,time:fmtTs()}]);
    addLog(`[ADMIN] Slot ${id} manually removed`,"sys");
  },[addLog]);

  const handleImageAnalysis = useCallback((result)=>{
    if(!result?.slots?.length) return;
    const cols=Math.ceil(Math.sqrt(result.slots.length));
    const newSlots={};
    result.slots.forEach((s,i)=>{
      const row=Math.floor(i/cols), col=i%cols;
      newSlots[s.id]={status:s.status,coords:[40+col*140,70+row*145,155+col*140,165+row*145],row:s.row||["A","B","C"][row]||"A",confidence:s.confidence||.85};
    });
    setSlots(newSlots);
    addLog(`[ADMIN] Map updated from image — ${result.slots.length} slots applied`,"img");
  },[addLog]);

  return(
    <div style={{minHeight:"100vh",background:C.bg,color:C.text,fontFamily:C.sans,paddingBottom:48}}>
      <style>{GLOBAL_STYLES}</style>
      <div style={{borderBottom:`1px solid ${C.border}`,background:"rgba(7,10,16,.92)",backdropFilter:"blur(16px)",padding:"0 20px",position:"sticky",top:0,zIndex:100}}>
        <div style={{maxWidth:960,margin:"0 auto",display:"flex",alignItems:"center",gap:16,height:58}}>
          <div style={{display:"flex",alignItems:"center",gap:10,marginRight:6}}>
            <div style={{width:34,height:34,borderRadius:10,background:"linear-gradient(135deg,#0ea5e9,#6366f1)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:16}}>🅿️</div>
            <div>
              <div style={{fontWeight:800,fontSize:14,letterSpacing:"-0.02em",lineHeight:1.1}}>CIT-U Parking</div>
              <div style={{fontSize:9,fontFamily:C.mono,color:C.muted,letterSpacing:"0.05em"}}>AI-POWERED · YOLO v5n</div>
            </div>
          </div>
          {[{id:"user",label:"🚗 Driver View"},{id:"admin",label:"🛡️ Admin Panel"}].map(t=>(
            <button key={t.id} onClick={()=>setTab(t.id)} style={{padding:"0 16px",height:58,border:"none",cursor:"pointer",fontFamily:C.sans,fontWeight:700,fontSize:13,background:"transparent",color:tab===t.id?t.id==="admin"?C.occ:C.accent:C.muted,borderBottom:tab===t.id?`2px solid ${t.id==="admin"?C.occ:C.accent}`:"2px solid transparent",transition:"all .2s"}}>{t.label}</button>
          ))}
          <div style={{marginLeft:"auto",display:"flex",alignItems:"center",gap:6}}>
            <LiveDot/><span style={{fontFamily:C.mono,fontSize:10,color:C.muted}}>LIVE · {fmtTs()}</span>
          </div>
        </div>
      </div>
      <div style={{maxWidth:960,margin:"0 auto",padding:"24px 16px"}}>
        {tab==="user"
          ?<UserView slots={slots}/>
          :<AdminPanel slots={slots} logs={logs} onRemove={handleRemove} removedSlots={removed} addLog={addLog} onImageAnalysis={handleImageAnalysis}/>}
      </div>
    </div>
  );
}