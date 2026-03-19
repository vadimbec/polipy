import { useState, useMemo } from "react";

const RAW = [
  {n:"Jean-Luc Mélenchon",p:"Professeur",g:"LFI",x:-0.5,y:6.5,m:8},
  {n:"Mathilde Panot",p:"Collab. parlementaire",g:"LFI",x:-0.5,y:5.5,m:2},
  {n:"Manuel Bompard",p:"Ingénieur",g:"LFI",x:2,y:7.5,m:2},
  {n:"Clémentine Autain",p:"Journaliste",g:"LFI",x:1,y:7,m:3},
  {n:"Alexis Corbière",p:"Professeur certifié",g:"LFI",x:-0.5,y:6.5,m:3},
  {n:"François Ruffin",p:"Journaliste",g:"LFI",x:1,y:7,m:2},
  {n:"Adrien Quatennens",p:"Attaché parlementaire",g:"LFI",x:-0.5,y:5.5,m:2},
  {n:"Danièle Obono",p:"Cadre administratif",g:"LFI",x:1.5,y:7,m:2},
  {n:"Eric Coquerel",p:"Permanent politique",g:"LFI",x:0,y:5,m:2},
  {n:"Raquel Garrido",p:"Avocate",g:"LFI",x:5,y:8.5,m:1},
  {n:"Antoine Léaument",p:"Collab. parlementaire",g:"LFI",x:-0.5,y:5.5,m:1},
  {n:"Louis Boyard",p:"Étudiant",g:"LFI",x:0,y:2,m:1},
  {n:"Sarah Legrain",p:"Professeur agrégé",g:"LFI",x:-0.5,y:7.5,m:1},
  {n:"Sophia Chikirou",p:"Conseil en communication",g:"LFI",x:4,y:7.5,m:1},
  {n:"Fabien Roussel",p:"Journaliste",g:"GDR",x:1,y:7,m:3},
  {n:"André Chassaigne",p:"Prof. des écoles",g:"GDR",x:-1,y:5.5,m:5},
  {n:"Sébastien Jumel",p:"Fonctionnaire",g:"GDR",x:0,y:5,m:2},
  {n:"Elsa Faucillon",p:"Professeur certifié",g:"GDR",x:-0.5,y:6.5,m:2},
  {n:"Boris Vallaud",p:"Haut fonctionnaire",g:"SOC",x:1,y:9,m:3},
  {n:"Valérie Rabault",p:"Cadre bancaire",g:"SOC",x:3,y:7,m:3},
  {n:"Jérôme Guedj",p:"Haut fonctionnaire",g:"SOC",x:1,y:9,m:3},
  {n:"C. Pirès Beaune",p:"Fonctionnaire",g:"SOC",x:0,y:5,m:3},
  {n:"Arthur Delaporte",p:"Collab. parlementaire",g:"SOC",x:-0.5,y:5.5,m:1},
  {n:"Olivier Faure",p:"Journaliste",g:"SOC",x:1,y:7,m:4},
  {n:"Cyrielle Chatelain",p:"Cadre",g:"ECO",x:2,y:7,m:1},
  {n:"Sandrine Rousseau",p:"Maître de conférences",g:"ECO",x:0,y:8,m:1},
  {n:"Julien Bayou",p:"Avocat",g:"ECO",x:5,y:8.5,m:1},
  {n:"Eva Sas",p:"Expert-comptable",g:"ECO",x:4,y:8,m:2},
  {n:"Sandra Regol",p:"Cadre administratif",g:"ECO",x:1.5,y:7,m:1},
  {n:"Gabriel Attal",p:"Haut fonctionnaire",g:"RE",x:1,y:9,m:3},
  {n:"Aurore Bergé",p:"Conseil en comm.",g:"RE",x:4,y:7.5,m:3},
  {n:"Yaël Braun-Pivet",p:"Magistrat",g:"RE",x:0.5,y:9,m:2},
  {n:"Stanislas Guerini",p:"Consultant",g:"RE",x:4,y:7.5,m:2},
  {n:"Olivia Grégoire",p:"Dirigeante de société",g:"RE",x:8,y:9,m:2},
  {n:"Clément Beaune",p:"Haut fonctionnaire",g:"RE",x:1,y:9,m:1},
  {n:"Roland Lescure",p:"Banquier",g:"RE",x:8,y:9,m:2},
  {n:"Sacha Houlié",p:"Avocat",g:"RE",x:5,y:8.5,m:2},
  {n:"Bruno Bonnell",p:"Chef d'entreprise",g:"RE",x:8,y:9,m:2},
  {n:"Maud Bregeon",p:"Ingénieur",g:"RE",x:2,y:7.5,m:1},
  {n:"P.-A. Anglade",p:"Cadre d'entreprise",g:"RE",x:2.5,y:7,m:2},
  {n:"Prisca Thevenot",p:"Cadre d'entreprise",g:"RE",x:2.5,y:7,m:1},
  {n:"J.-R. Cazeneuve",p:"Chef d'entreprise",g:"RE",x:8,y:9,m:2},
  {n:"Marc Ferracci",p:"Prof. d'université",g:"RE",x:0,y:9,m:1},
  {n:"Marie Lebec",p:"Cadre",g:"RE",x:2,y:7,m:2},
  {n:"François Bayrou",p:"Professeur agrégé",g:"DEM",x:-0.5,y:7.5,m:10},
  {n:"Jean-Paul Mattei",p:"Avocat",g:"DEM",x:5,y:8.5,m:2},
  {n:"Erwan Balanant",p:"Avocat",g:"DEM",x:5,y:8.5,m:2},
  {n:"Bruno Millienne",p:"Commerçant",g:"DEM",x:3,y:3.5,m:2},
  {n:"Édouard Philippe",p:"Haut fonctionnaire",g:"HOR",x:1,y:9,m:4},
  {n:"L. Marcangeli",p:"Avocat",g:"HOR",x:5,y:8.5,m:3},
  {n:"Naïma Moutchou",p:"Avocate",g:"HOR",x:5,y:8.5,m:2},
  {n:"Laurent Wauquiez",p:"Haut fonctionnaire",g:"LR",x:1,y:9,m:5},
  {n:"Éric Ciotti",p:"Fonctionnaire",g:"LR",x:0,y:5,m:5},
  {n:"Bruno Retailleau",p:"Cadre d'entreprise",g:"LR",x:2.5,y:7,m:6},
  {n:"Aurélien Pradié",p:"Attaché parlementaire",g:"LR",x:-0.5,y:5.5,m:2},
  {n:"P.-H. Dumont",p:"Avocat",g:"LR",x:5,y:8.5,m:2},
  {n:"V. Duby-Muller",p:"Cadre",g:"LR",x:2,y:7,m:3},
  {n:"Olivier Marleix",p:"Haut fonctionnaire",g:"LR",x:1,y:9,m:3},
  {n:"Annie Genevard",p:"Professeur certifié",g:"LR",x:-0.5,y:6.5,m:3},
  {n:"R. Schellenberger",p:"Attaché parlementaire",g:"LR",x:-0.5,y:5.5,m:2},
  {n:"Marine Le Pen",p:"Avocate",g:"RN",x:5,y:8.5,m:6},
  {n:"Jordan Bardella",p:"Permanent politique",g:"RN",x:0,y:5,m:2},
  {n:"Sébastien Chenu",p:"Conseil en comm.",g:"RN",x:4,y:7.5,m:2},
  {n:"Laure Lavalette",p:"Collab. parlementaire",g:"RN",x:-0.5,y:5.5,m:1},
  {n:"Laurent Jacobelli",p:"Consultant",g:"RN",x:4,y:7.5,m:1},
  {n:"Edwige Diaz",p:"Agent territorial",g:"RN",x:-2,y:3.5,m:1},
  {n:"Thomas Ménagé",p:"Cadre",g:"RN",x:2,y:7,m:1},
  {n:"Julien Odoul",p:"Cadre d'entreprise",g:"RN",x:2.5,y:7,m:1},
  {n:"G. de Fournas",p:"Exploitant agricole",g:"RN",x:4,y:4,m:1},
  {n:"Franck Allisio",p:"Chef d'entreprise",g:"RN",x:8,y:9,m:1},
  {n:"H. de Lépinau",p:"Avocat",g:"RN",x:5,y:8.5,m:1},
  {n:"Kévin Mauvieux",p:"Ouvrier qualifié",g:"RN",x:-5,y:2.5,m:1},
  {n:"Jocelyn Dessigny",p:"Artisan",g:"RN",x:3,y:3.5,m:1},
  {n:"Stéphanie Galzy",p:"Professeur certifié",g:"RN",x:-0.5,y:6.5,m:1},
  {n:"Christophe Bentz",p:"Avocat",g:"RN",x:5,y:8.5,m:1},
  {n:"Bryan Masson",p:"Employé",g:"RN",x:-3,y:3,m:1},
  {n:"Angélique Ranc",p:"Infirmière",g:"RN",x:-1.5,y:5,m:1},
  {n:"José Gonzalez",p:"Retraité",g:"RN",x:1,y:4,m:1},
  {n:"N. Dupont-Aignan",p:"Haut fonctionnaire",g:"NI",x:1,y:9,m:6},
];

const GM = {
  LFI:{l:"La France Insoumise",c:"#DC2626",o:0},
  GDR:{l:"GDR (PCF)",c:"#7F1D1D",o:1},
  SOC:{l:"Socialistes",c:"#EC4899",o:2},
  ECO:{l:"Écologistes",c:"#16A34A",o:3},
  RE:{l:"Renaissance",c:"#F59E0B",o:4},
  DEM:{l:"MoDem",c:"#FB923C",o:5},
  HOR:{l:"Horizons",c:"#38BDF8",o:6},
  LR:{l:"Les Républicains",c:"#1E40AF",o:7},
  RN:{l:"Rass. National",c:"#1E293B",o:8},
  NI:{l:"Non-inscrits",c:"#6B7280",o:9},
};

function jit(v,a=0.2){return v+(Math.random()-0.5)*a*2}

export default function App() {
  const [hov, setHov] = useState(null);
  const [act, setAct] = useState(new Set(Object.keys(GM)));
  const [zones, setZones] = useState(true);

  const W=820, H=720;
  const mg={t:55,r:25,b:75,l:65};
  const pW=W-mg.l-mg.r, pH=H-mg.t-mg.b;
  const xR=[-8,10], yR=[0,10.5];
  const sx=v=>mg.l+((v-xR[0])/(xR[1]-xR[0]))*pW;
  const sy=v=>mg.t+pH-((v-yR[0])/(yR[1]-yR[0]))*pH;

  const data=useMemo(()=>RAW.map((d,i)=>({...d,id:i,jx:jit(d.x,0.3),jy:jit(d.y,0.25),col:GM[d.g]?.c||"#999",lab:GM[d.g]?.l||d.g})),[]);
  const filt=data.filter(d=>act.has(d.g));

  const bary=useMemo(()=>{
    const g={};
    filt.forEach(d=>{if(!g[d.g])g[d.g]={sx:0,sy:0,n:0};g[d.g].sx+=d.x;g[d.g].sy+=d.y;g[d.g].n++});
    return Object.entries(g).map(([k,v])=>({g:k,x:v.sx/v.n,y:v.sy/v.n,n:v.n,c:GM[k]?.c,l:GM[k]?.l}));
  },[filt]);

  const tog=g=>{setAct(p=>{const n=new Set(p);n.has(g)?n.delete(g):n.add(g);return n})};

  const hd=hov!==null?data[hov]:null;

  return(
    <div style={{fontFamily:"'Helvetica Neue',system-ui,sans-serif",background:"#FAF9F7",minHeight:"100vh",padding:16,color:"#1a1a1a"}}>
      <div style={{maxWidth:W,margin:"0 auto 12px",textAlign:"center"}}>
        <h1 style={{fontSize:26,fontWeight:900,margin:"0 0 2px",letterSpacing:"-0.5px"}}>L'Assemblée des Classes</h1>
        <p style={{fontSize:12,color:"#8C8C8C",margin:0,fontWeight:400}}>Position sociale des députés — Axes exploitation (Marx) × domination (Bourdieu) — Prototype ~80 députés</p>
      </div>

      <div style={{maxWidth:W,margin:"0 auto 10px",display:"flex",flexWrap:"wrap",gap:5,justifyContent:"center"}}>
        {Object.entries(GM).sort((a,b)=>a[1].o-b[1].o).map(([g,m])=>(
          <button key={g} onClick={()=>tog(g)} style={{
            display:"flex",alignItems:"center",gap:4,padding:"2px 10px",borderRadius:20,
            border:`2px solid ${m.c}`,background:act.has(g)?m.c:"transparent",
            color:act.has(g)?"#fff":m.c,cursor:"pointer",fontSize:10.5,fontWeight:700,
            fontFamily:"inherit",opacity:act.has(g)?1:0.35,transition:"all 0.15s"
          }}>{m.l}</button>
        ))}
        <button onClick={()=>setZones(p=>!p)} style={{
          padding:"2px 10px",borderRadius:20,border:"1px solid #ccc",background:zones?"#eee":"transparent",
          color:"#666",cursor:"pointer",fontSize:10,fontFamily:"inherit"
        }}>{zones?"☑":"☐"} Zones</button>
      </div>

      <div style={{maxWidth:W,margin:"0 auto",position:"relative"}}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{width:"100%",height:"auto",display:"block"}}>
          <rect x={mg.l} y={mg.t} width={pW} height={pH} fill="#FEFDFB" stroke="#DDD" strokeWidth={1}/>

          {zones&&<>
            <rect x={sx(-8)} y={sy(5)} width={sx(0)-sx(-8)} height={sy(0)-sy(5)} fill="rgba(220,38,38,0.035)"/>
            <rect x={sx(0)} y={sy(10.5)} width={sx(10)-sx(0)} height={sy(5)-sy(10.5)} fill="rgba(37,99,235,0.035)"/>
            <rect x={sx(0)} y={sy(5)} width={sx(10)-sx(0)} height={sy(0)-sy(5)} fill="rgba(245,158,11,0.03)"/>
            <rect x={sx(-8)} y={sy(10.5)} width={sx(0)-sx(-8)} height={sy(5)-sy(10.5)} fill="rgba(34,197,94,0.025)"/>
            {[{t:"PROLÉTARIAT",x:-4.5,y:1.8},{t:"PETITE BOURGEOISIE",x:5.5,y:1.8},{t:"BOURGEOISIE",x:7,y:9.5},{t:"FONCTIONNAIRES",x:-3,y:7.5}].map((z,i)=>
              <text key={i} x={sx(z.x)} y={sy(z.y)} textAnchor="middle" fontSize={9} fontWeight={700} fill="#BBB" style={{pointerEvents:"none"}}>{z.t}</text>
            )}
          </>}

          {[-6,-4,-2,0,2,4,6,8].map(v=><line key={`gx${v}`} x1={sx(v)} y1={mg.t} x2={sx(v)} y2={mg.t+pH} stroke={v===0?"#AAA":"#E8E8E8"} strokeWidth={v===0?1.5:0.5} strokeDasharray={v===0?"":"2,3"}/>)}
          {[1,2,3,4,5,6,7,8,9,10].map(v=><line key={`gy${v}`} x1={mg.l} y1={sy(v)} x2={mg.l+pW} y2={sy(v)} stroke={v===5?"#AAA":"#E8E8E8"} strokeWidth={v===5?1.5:0.5} strokeDasharray={v===5?"":"2,3"}/>)}

          {[-6,-4,-2,0,2,4,6,8].map(v=><text key={`tx${v}`} x={sx(v)} y={mg.t+pH+16} textAnchor="middle" fontSize={9} fill="#AAA">{v>0?`+${v}`:v}</text>)}
          {[0,2,4,6,8,10].map(v=><text key={`ty${v}`} x={mg.l-8} y={sy(v)+3} textAnchor="end" fontSize={9} fill="#AAA">{v}</text>)}

          <text x={sx(1)} y={H-10} textAnchor="middle" fontSize={11.5} fontWeight={800} fill="#666">
            ← TRAVAIL ————— Axe Marx (Exploitation) ————— CAPITAL →
          </text>
          <text transform={`translate(14,${mg.t+pH/2}) rotate(-90)`} textAnchor="middle" fontSize={11.5} fontWeight={800} fill="#666">
            ← DOMINÉ ————— Axe Bourdieu (Domination) ————— DOMINANT →
          </text>

          {bary.filter(b=>b.n>1).map(b=>{
            const cx=sx(b.x),cy=sy(b.y);
            return(<g key={`b${b.g}`} opacity={0.5}>
              <line x1={cx-7} y1={cy} x2={cx+7} y2={cy} stroke={b.c} strokeWidth={2.5}/>
              <line x1={cx} y1={cy-7} x2={cx} y2={cy+7} stroke={b.c} strokeWidth={2.5}/>
              <circle cx={cx} cy={cy} r={11} fill="none" stroke={b.c} strokeWidth={1.5} strokeDasharray="3,2"/>
            </g>)
          })}

          {filt.map(d=>{
            const cx=sx(d.jx),cy=sy(d.jy),r=3.5+Math.min(d.m,8)*0.7,h=hov===d.id;
            return(<g key={d.id} onMouseEnter={()=>setHov(d.id)} onMouseLeave={()=>setHov(null)} style={{cursor:"pointer"}}>
              <circle cx={cx} cy={cy} r={h?r+4:r} fill={d.col} fillOpacity={h?1:0.7} stroke={h?"#fff":d.col} strokeWidth={h?2.5:0.8} style={{transition:"all 0.12s"}}/>
              {h&&<text x={cx} y={cy-r-7} textAnchor="middle" fontSize={9.5} fontWeight={800} fill="#1a1a1a" style={{pointerEvents:"none"}}>{d.n}</text>}
            </g>)
          })}
        </svg>

        {hd&&<div style={{position:"absolute",top:6,right:6,background:"rgba(255,255,255,0.97)",border:`2px solid ${hd.col}`,borderRadius:8,padding:"10px 14px",fontSize:11.5,lineHeight:1.6,boxShadow:"0 4px 24px rgba(0,0,0,0.1)",pointerEvents:"none",zIndex:10,maxWidth:210}}>
          <div style={{fontWeight:900,fontSize:14,marginBottom:2}}>{hd.n}</div>
          <div style={{color:hd.col,fontWeight:700,fontSize:10.5,marginBottom:5}}>{hd.lab}</div>
          <div><span style={{color:"#999"}}>Profession :</span> {hd.p}</div>
          <div><span style={{color:"#999"}}>Exploitation :</span> {hd.x>0?`+${hd.x}`:hd.x} <span style={{fontSize:9,color:"#BBB"}}>{hd.x>3?"(capital)":hd.x<-2?"(travail)":"(interméd.)"}</span></div>
          <div><span style={{color:"#999"}}>Domination :</span> {hd.y}/10 <span style={{fontSize:9,color:"#BBB"}}>{hd.y>=8?"(dominant)":hd.y<=4?"(dominé)":"(interméd.)"}</span></div>
          <div><span style={{color:"#999"}}>Mandats :</span> {hd.m}</div>
        </div>}
      </div>

      <div style={{maxWidth:W,margin:"14px auto 0",display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:8}}>
        {[
          {l:"Position moy. X (Exploitation)",v:"+1.9",s:"L'Assemblée penche côté capital"},
          {l:"Position moy. Y (Domination)",v:"7.0 / 10",s:"Massivement dans les dominants"},
          {l:"Profession la + fréquente",v:"Avocat·e",s:"~18% de l'échantillon"},
        ].map((s,i)=>(<div key={i} style={{background:"#fff",border:"1px solid #E5E5E5",borderRadius:8,padding:10,textAlign:"center"}}>
          <div style={{fontSize:9,color:"#999",fontWeight:700,textTransform:"uppercase",letterSpacing:"0.5px"}}>{s.l}</div>
          <div style={{fontSize:20,fontWeight:900,margin:"3px 0 1px"}}>{s.v}</div>
          <div style={{fontSize:9.5,color:"#BBB"}}>{s.s}</div>
        </div>))}
      </div>

      <p style={{maxWidth:W,margin:"10px auto 0",fontSize:9.5,color:"#BBB",textAlign:"center",lineHeight:1.5}}>
        Prototype Phase 1 · ~80 députés · Axe X = profession d'origine (proxy rapport au capital) · Axe Y = position hiérarchie PCS · ⊕ = barycentre du groupe<br/>
        Sources : NosDéputés.fr (Regards Citoyens) · Inspiré du graphique de Positions Revue (Chris / @pasduhring)
      </p>
    </div>
  );
}
