// The simulations page. It fetches the sources (sources.json), a source's
// index (index/<source>.json, one column per setting or statistic) and one
// simulation's detail (sim/<source>/<id>.json), all relative to the page,
// so it works under the monitor's /sims/ and on its own. Filters: levels of
// a factor are alternatives, factors and bins all hold at once. The hash
// keeps the source and the simulation open in the deep dive.
(function(){
"use strict";
var PAL=window.simsPalette,LAYERS=window.simsLayers,MAXCOL=PAL.length-1;
var NS="http://www.w3.org/2000/svg";
var S={layer:null,src:null,data:null,shown:[],levels:{},bins:{},group:null,
 scale:"log",norm:false,sort:"cases",desc:true,sim:null};
var STATS={cases:"Cases",strata:"Strata",days:"Days",weeks:"Weeks",
 peak:"Peak week's cases",peak_at:"Peak position",delay:"Mean delay (days)",
 within7:"Share within 7 days",negative:"Share of negative updates",
 within14:"Share within 14 days",within28:"Share within 28 days",
 releases:"Releases a week",gap:"Days between releases",first_report:"Days to the first report",
 growth:"Largest weekly growth (log)",noise:"Observation noise"};
function $(id){return document.getElementById(id)}
function esc(t){return String(t).replace(/[&<>"]/g,function(c){
 return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]})}
function fmt(v){if(v===null||v===undefined||isNaN(v))return "–";
 var a=Math.abs(v);if(a>=1e4)return Math.round(v).toLocaleString();
 if(a>=100)return String(Math.round(v));if(a>=1)return v.toFixed(2).replace(/\.?0+$/,"");
 if(a===0)return "0";return v.toPrecision(2)}
function median(a){var b=a.filter(function(v){return v!==null&&!isNaN(v)})
 .sort(function(x,y){return x-y});if(!b.length)return null;
 var m=Math.floor(b.length/2);return b.length%2?b[m]:(b[m-1]+b[m])/2}
function width(el){return Math.max(260,el.clientWidth||440)}

// ---- tooltip -----------------------------------------------------------
var tip=$("tip");
function showTip(e,html){tip.innerHTML=html;tip.hidden=false;
 var x=e.clientX+12,y=e.clientY+12;
 if(x+290>window.innerWidth)x=e.clientX-290;tip.style.left=x+"px";tip.style.top=y+"px"}
function hideTip(){tip.hidden=true}

// ---- scales and charts -------------------------------------------------
function ticks(lo,hi,n){if(!(hi>lo))return [lo];var step=Math.pow(10,Math.floor(Math.log10((hi-lo)/n)));
 var err=(hi-lo)/n/step;if(err>=5)step*=5;else if(err>=2)step*=2;
 var out=[];for(var v=Math.ceil(lo/step)*step;v<=hi+1e-9;v+=step)out.push(+v.toFixed(10));return out}
// dateTicks: day numbers a whole number of weeks apart, at most n of them.
function dateTicks(lo,hi,n){var steps=[7,14,28,56,91,182,364,728],st=steps[steps.length-1];
 for(var k=0;k<steps.length;k++)if((hi-lo)/steps[k]<=n){st=steps[k];break}
 var out=[];for(var v=lo;v<=hi;v+=st)out.push(v);return out}
function logTicks(lo,hi){var out=[];for(var p=Math.floor(Math.log10(lo));p<=Math.ceil(Math.log10(hi));p++){
 var v=Math.pow(10,p);if(v>=lo&&v<=hi)out.push(v)}return out.length?out:[lo,hi]}
// lineChart: series [{x:[], y:[], colour, w, op, id, name}] on one panel,
// x numeric (or dates as day numbers with `dates`), a log or linear y. With
// `pick` hovering shows the nearest series and clicking calls pick(id).
function lineChart(el,o){
 var W=width(el),H=o.height||220,L=52,R=10,T=8,B=30,pw=W-L-R,ph=H-T-B;
 var xs=[],ys=[];o.series.forEach(function(s){s.x.forEach(function(x,i){
  var y=s.y[i];if(y===null||isNaN(y))return;if(o.log&&y<=0)return;xs.push(x);ys.push(y)})});
 if(!xs.length){el.innerHTML="<p class=mut>Nothing to show.</p>";return}
 var x0=Math.min.apply(null,xs),x1=Math.max.apply(null,xs);if(x1===x0)x1=x0+1;
 var y0=o.log?Math.min.apply(null,ys):Math.min(0,Math.min.apply(null,ys)),y1=Math.max.apply(null,ys);
 if(o.ymin!==undefined)y0=Math.min(y0,o.ymin);if(y1===y0)y1=y0+1;
 var fx=function(x){return L+(x-x0)/(x1-x0)*pw};
 var fy=o.log?function(y){return T+ph-(Math.log(y)-Math.log(y0))/(Math.log(y1)-Math.log(y0))*ph}
  :function(y){return T+ph-(y-y0)/(y1-y0)*ph};
 var g=[];
 (o.log?logTicks(y0,y1):ticks(y0,y1,4)).forEach(function(v){var y=fy(v);
  g.push('<line class=grid x1='+L+' x2='+(W-R)+' y1='+y+' y2='+y+'></line><text class=ax x='+(L-4)+' y='+(y+4)+' text-anchor=end>'+fmt(v)+'</text>')});
 var xt=o.dates?dateTicks(x0,x1,Math.max(2,Math.floor(pw/90))):ticks(x0,x1,Math.max(2,Math.floor(pw/80)));
 xt.forEach(function(v){var x=fx(v);g.push('<text class=ax x='+x+' y='+(H-B+14)+' text-anchor='+(x>W-R-40?'end':'middle')+'>'+
  (o.dates?o.dates(v):fmt(v))+'</text>')});
 if(o.xlab)g.push('<text class=ax x='+(L+pw/2)+' y='+(H-2)+' text-anchor=middle>'+esc(o.xlab)+'</text>');
 if(o.ylab)g.push('<text class=ax transform="rotate(-90)" x='+(-(T+ph/2))+' y=11 text-anchor=middle>'+esc(o.ylab)+'</text>');
 if(o.hline!==undefined){var hy=fy(o.hline);g.push('<line class=mark x1='+L+' x2='+(W-R)+' y1='+hy+' y2='+hy+'></line>')}
 var pts=[];
 o.series.forEach(function(s,si){var d="",pen=false;s.x.forEach(function(x,i){var y=s.y[i];
  if(y===null||isNaN(y)||(o.log&&y<=0)){pen=false;return}
  var px=fx(x),py=fy(y);d+=(pen?"L":"M")+px.toFixed(1)+" "+py.toFixed(1);pen=true;
  if(o.pick||o.hover)pts.push([px,py,si,i])});
  g.push('<path d="'+d+'" fill=none stroke="'+s.colour+'" stroke-width='+(s.w||2)+
   ' stroke-opacity='+(s.op===undefined?1:s.op)+(s.dash?' stroke-dasharray="'+s.dash+'"':'')+' stroke-linejoin=round></path>')});
 el.innerHTML='<svg width='+W+' height='+H+' viewBox="0 0 '+W+' '+H+'">'+g.join("")+
  '<circle id=dot r=4 fill=none stroke-width=2 visibility=hidden></circle>'+
  '<rect x='+L+' y='+T+' width='+pw+' height='+ph+' fill=transparent></rect></svg>';
 if(!(o.pick||o.hover))return;
 var svg=el.querySelector("svg"),dot=svg.querySelector("#dot"),cur=null;
 svg.addEventListener("mousemove",function(e){var r=svg.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,best=null,bd=900;
  pts.forEach(function(p){var d=(p[0]-mx)*(p[0]-mx)+(p[1]-my)*(p[1]-my);if(d<bd){bd=d;best=p}});
  if(!best){cur=null;dot.setAttribute("visibility","hidden");hideTip();svg.style.cursor="";return}
  cur=best;var s=o.series[best[2]];dot.setAttribute("cx",best[0]);dot.setAttribute("cy",best[1]);
  dot.setAttribute("stroke",s.colour);dot.setAttribute("visibility","visible");
  svg.style.cursor=o.pick?"pointer":"";
  showTip(e,(o.hover||defaultHover)(s,best[3],o))});
 svg.addEventListener("mouseleave",function(){dot.setAttribute("visibility","hidden");hideTip()});
 if(o.pick)svg.addEventListener("click",function(){if(cur)o.pick(o.series[cur[2]].id)})}
function defaultHover(s,i,o){return "<b>"+esc(s.name)+"</b><br>"+(o.dates?o.dates(s.x[i]):fmt(s.x[i]))+": "+fmt(s.y[i])}

// The layer panels share one size (PW x PH), so the grid is even: shares
// of each level (shown solid over all faint) and histograms (all faint,
// shown solid), on the same colours.
var PW=240,PH=132;
function panelSvg(inner){return '<svg viewBox="0 0 '+PW+' '+PH+'" width="100%" role="img">'+inner+'</svg>'}
// barsH: one row per level, its share of the shown simulations solid over
// its share of the source's faint; click a level to filter to it.
function barsH(el,rows,click){
 var n=rows.length,lab=86,R=34,rh=Math.min(18,(PH-4)/n),pw=PW-lab-R,g=[];
 var tall=rows.reduce(function(a,r){return a+r.all},0)||1,tshown=rows.reduce(function(a,r){return a+r.n},0)||1;
 // Shares drawn against the largest, so a panel's bars fill its width.
 var top=Math.max.apply(null,rows.map(function(r){return Math.max(r.all/tall,r.n/tshown)}))||1;
 rows.forEach(function(r,i){var y=2+i*rh,sa=r.all/tall/top,ss=r.n/tshown/top,bh=Math.max(3,rh-5);
  g.push('<g class=bar data-i='+i+'><text class=ax x='+(lab-4)+' y='+(y+rh/2+4)+' text-anchor=end'+(r.on?' style="fill:var(--c2);font-weight:600"':'')+'>'+esc(r.label.length>14?r.label.slice(0,13)+"…":r.label)+'</text>'+
   '<rect x='+lab+' y='+(y+(rh-bh)/2)+' width='+(sa*pw)+' height='+bh+' rx=2 fill="var(--c1)" fill-opacity=.2></rect>'+
   '<rect x='+lab+' y='+(y+(rh-bh)/2+bh*0.25)+' width='+(ss*pw)+' height='+(bh*0.5)+' rx=2 fill="'+(r.on?"var(--c2)":"var(--c1)")+'"></rect>'+
   '<text class=ax x='+(lab+Math.max(sa,ss)*pw+3)+' y='+(y+rh/2+4)+'>'+Math.round(100*r.n/tshown)+'%</text>'+
   '<rect class=hit x=0 y='+y+' width='+PW+' height='+rh+'></rect></g>')});
 el.innerHTML=panelSvg(g.join(""));
 [].forEach.call(el.querySelectorAll("g.bar"),function(b){var r=rows[+b.dataset.i];
  b.addEventListener("click",function(){click(r)});
  b.addEventListener("mousemove",function(e){showTip(e,"<b>"+esc(r.label)+"</b><br>"+r.n+" shown ("+Math.round(100*r.n/tshown)+"%), "+r.all+" in the source ("+Math.round(100*r.all/tall)+"%)")});
  b.addEventListener("mouseleave",hideTip)})}

// hist: 20 bins over the source's range, on a log scale for positive
// values over more than three orders of magnitude with a long right tail;
// all faint, shown solid, each scaled to its own peak so the shapes
// compare; click a bin to filter to it.
function hist(el,all,shown,sel,click){
 var a=all.filter(function(v){return v!==null}),L=6,R=6,T=6,B=18,pw=PW-L-R,ph=PH-T-B;
 if(!a.length){el.innerHTML="<p class=mut>No values.</p>";return}
 var mn=Math.min.apply(null,a),mxv=Math.max.apply(null,a),md=median(a),lg=mn>0&&mxv/mn>1000&&mxv-md>10*(md-mn);
 var tf=lg?Math.log10:function(v){return v},inv=lg?function(v){return Math.pow(10,v)}:function(v){return v};
 var lo=tf(mn),hi=tf(mxv);if(hi===lo)hi=lo+1;var n=20,bw=(hi-lo)/n;
 function count(v){var c=new Array(n).fill(0);v.forEach(function(x){if(x===null||(lg&&x<=0))return;c[Math.min(n-1,Math.floor((tf(x)-lo)/bw))]++});return c}
 var ca=count(a),cs=count(shown),ma=Math.max.apply(null,ca)||1,ms=Math.max.apply(null,cs)||1,g=[],gap=1.5,w=pw/n;
 g.push('<line class=grid x1='+L+' x2='+(PW-R)+' y1='+(T+ph)+' y2='+(T+ph)+'></line>');
 for(var i=0;i<n;i++){var x=L+i*w,on=sel&&Math.abs(sel[0]-inv(lo+i*bw))<=Math.abs(inv(lo+i*bw))*1e-9+1e-12;
  g.push('<g class=bar data-i='+i+'><rect x='+(x+gap/2)+' y='+(T+ph-ca[i]/ma*ph)+' width='+(w-gap)+' height='+(ca[i]/ma*ph)+' fill="var(--c1)" fill-opacity=.2></rect>'+
   '<rect x='+(x+gap/2+w*0.2)+' y='+(T+ph-cs[i]/ms*ph)+' width='+Math.max(1,w*0.6-gap)+' height='+(cs[i]/ms*ph)+' fill="'+(on?"var(--c2)":"var(--c1)")+'"></rect>'+
   '<rect class=hit x='+x+' y='+T+' width='+w+' height='+ph+'></rect></g>')}
 [0,n].forEach(function(k){g.push('<text class=ax x='+(L+k*w)+' y='+(PH-4)+' text-anchor='+(k?'end':'start')+'>'+fmt(inv(lo+k*bw))+'</text>')});
 if(lg)g.push('<text class=ax x='+(PW/2)+' y='+(PH-4)+' text-anchor=middle>log scale</text>');
 el.innerHTML=panelSvg(g.join(""));
 [].forEach.call(el.querySelectorAll("g.bar"),function(b){var i=+b.dataset.i,r=[inv(lo+i*bw),inv(lo+(i+1)*bw)];
  b.addEventListener("click",function(){click(r)});
  b.addEventListener("mousemove",function(e){showTip(e,fmt(r[0])+" to "+fmt(r[1])+"<br>"+cs[i]+" shown, "+ca[i]+" in the source")});
  b.addEventListener("mouseleave",hideTip)})}

// heat: a matrix in one sequential hue (the first accent), by its share
// of the largest cell on a log scale.
function heat(el,m,xl,yl){
 var rows=m.length,cols=m[0]?m[0].length:0,W=width(el),L=44,T=6,B=30,cw=Math.max(4,Math.min(26,(W-L-8)/Math.max(cols,1))),
  ch=Math.max(3,Math.min(14,260/Math.max(rows,1))),H=T+rows*ch+B,mx=0,g=[];
 m.forEach(function(r){r.forEach(function(v){if(v>mx)mx=v})});
 m.forEach(function(r,i){r.forEach(function(v,j){if(v<=0)return;var s=Math.log1p(v)/Math.log1p(mx);
  g.push('<rect x='+(L+j*cw)+' y='+(T+i*ch)+' width='+(cw-1)+' height='+(ch-1)+' fill="var(--c1)" fill-opacity='+(0.08+0.92*s).toFixed(3)+'><title>'+yl+' '+i+', '+xl+' '+j+': '+v+'</title></rect>')})});
 ticks(0,cols-1,Math.max(2,Math.floor(cols*cw/60))).forEach(function(j){g.push('<text class=ax x='+(L+j*cw+cw/2)+' y='+(T+rows*ch+12)+' text-anchor=middle>'+j+'</text>')});
 ticks(0,rows-1,5).forEach(function(i){g.push('<text class=ax x='+(L-4)+' y='+(T+i*ch+ch)+' text-anchor=end>'+i+'</text>')});
 g.push('<text class=ax x='+(L+cols*cw/2)+' y='+(H-2)+' text-anchor=middle>'+esc(xl)+'</text>');
 el.innerHTML='<svg width='+Math.max(W,L+cols*cw+8)+' height='+H+'>'+g.join("")+'</svg>'}

// ---- data, filters and groups -----------------------------------------
function factorNames(){return Object.keys(S.data.factors)}
function level(name,i){var f=S.data.factors[name];return f.levels[f.codes[i]]}
function filtered(){var d=S.data,out=[],n=d.ids.length;
 for(var i=0;i<n;i++){var ok=true;
  for(var k in S.levels){var set=S.levels[k];if(set.size&&!set.has(level(k,i))){ok=false;break}}
  if(ok)for(var b in S.bins){var v=d.values[b]?d.values[b][i]:d.stats[b][i],r=S.bins[b];
   if(v===null||v<r[0]||v>r[1]+1e-12){ok=false;break}}
  if(ok)out.push(i)}
 return out}
// groups: each level of the group-by factor its colour, in the level
// order; with more levels than colours, the source's most common keep
// theirs and the rest are "other". Colours follow the level, whatever
// the filters.
function groups(){var name=S.group;if(!name||!S.data.factors[name])return {of:function(){return "all"},
 colour:function(){return PAL[0]},list:["all"]};
 var f=S.data.factors[name],lv=f.levels,all=new Array(lv.length).fill(0);f.codes.forEach(function(c){all[c]++});
 var keep=lv;
 if(lv.length>PAL.length){var top=lv.map(function(_,j){return j}).sort(function(a,b){return all[b]-all[a]}).slice(0,MAXCOL);
  keep=lv.filter(function(_,j){return top.indexOf(j)>=0})}
 var col={};keep.forEach(function(l,j){col[l]=PAL[j]});
 var counts={};S.shown.forEach(function(i){var l=level(name,i);counts[l]=(counts[l]||0)+1});
 var kept=new Set(keep),list=keep.filter(function(l){return counts[l]});
 var other=Object.keys(counts).some(function(l){return !kept.has(l)});if(other)list.push("other");
 return {of:function(i){var l=level(name,i);return kept.has(l)?l:"other"},
  colour:function(l){return l==="other"?"var(--mut)":col[l]},list:list}}

function chips(){var h=[];Object.keys(S.levels).forEach(function(k){S.levels[k].forEach(function(l){
 h.push('<span class=chip data-k="'+esc(k)+'" data-l="'+esc(l)+'">'+esc(k)+": "+esc(l)+" ×</span>")})});
 Object.keys(S.bins).forEach(function(k){var r=S.bins[k];h.push('<span class=chip data-b="'+esc(k)+'">'+esc(k)+": "+fmt(r[0])+" to "+fmt(r[1])+" ×</span>")});
 $("chips").innerHTML=h.join("");
 [].forEach.call($("chips").querySelectorAll(".chip"),function(c){c.addEventListener("click",function(){
  if(c.dataset.b)delete S.bins[c.dataset.b];else toggleLevel(c.dataset.k,c.dataset.l,true);update()})})}
function toggleLevel(k,l,quiet){var s=S.levels[k]||(S.levels[k]=new Set());if(s.has(l))s.delete(l);else s.add(l);
 if(!s.size)delete S.levels[k];if(!quiet)update()}

// ---- sections -----------------------------------------------------------
function summary(){var d=S.data,sh=S.shown;
 function med(k){return median(sh.map(function(i){return d.stats[k][i]}))}
 var neg=sh.filter(function(i){return d.stats.negative[i]>0}).length;
 var t=[["Simulations",sh.length.toLocaleString(),"of "+d.ids.length.toLocaleString()+" in "+d.batches+" batches"],
  ["Median cases",fmt(med("cases")),""],["Median strata",fmt(med("strata")),""],
  ["Median weeks",fmt(med("weeks")),""],["Median mean delay",fmt(med("delay")),"days"],
  ["Median within 7 days",fmt(med("within7")),"share of cases"],
  ["With negative updates",sh.length?fmt(neg/sh.length):"–","share of simulations"]];
 $("tiles").innerHTML=t.map(function(x){return '<div class=tile><div class=k>'+x[0]+'</div><div class=v>'+x[1]+'</div><div class=s>'+x[2]+'</div></div>'}).join("");
 var g=groups(),by={};sh.forEach(function(i){var l=g.of(i);(by[l]=by[l]||[]).push(i)});
 $("bylabel").textContent=S.group||"nothing (choose a factor above)";
 var cols=["cases","strata","weeks","peak_at","delay","within7"];
 var h='<tr><th>Level</th><th>Simulations</th>'+cols.map(function(c){return "<th>Median "+STATS[c].toLowerCase()+"</th>"}).join("")+"</tr>";
 g.list.forEach(function(l){var ix=by[l]||[];h+='<tr data-l="'+esc(l)+'"><td><i class=sw style="background:'+g.colour(l)+'"></i>'+esc(l)+"</td><td>"+ix.length+"</td>"+
  cols.map(function(c){return "<td>"+fmt(median(ix.map(function(i){return d.stats[c][i]})))+"</td>"}).join("")+"</tr>"});
 $("bygroup").innerHTML=h;
 [].forEach.call($("bygroup").querySelectorAll("tr[data-l]"),function(r){r.addEventListener("click",function(){
  if(S.group&&r.dataset.l!=="other"&&r.dataset.l!=="all")toggleLevel(S.group,r.dataset.l)})})}

// The layers, one at a time (S.layer, the first by default): its settings
// that vary in the source as an even grid of panels in the grid's order,
// each named plainly with its grid note as its tooltip; those constant in
// the source on one line.
function plain(k){var t=(S.layer==="outcomes"&&STATS[k]||k).replace(/_/g," ");return t.charAt(0).toUpperCase()+t.slice(1)}
function settingsOf(layer){var d=S.data;
 if(layer==="outcomes")return Object.keys(d.stats);
 var mine=function(k){return d.layers[k]===layer&&(k in d.factors||k in d.values)};
 var ordered=(d.order||[]).filter(mine);
 return ordered.concat(Object.keys(d.layers).filter(function(k){return mine(k)&&ordered.indexOf(k)<0}))}
function varies(k){var d=S.data;
 if(k in d.factors)return d.factors[k].levels.length>1;
 var col=(d.values[k]||d.stats[k]).filter(function(v){return v!==null});
 return col.length>0&&Math.min.apply(null,col)<Math.max.apply(null,col)}
function constantText(k){var d=S.data;
 if(k in d.factors)return d.factors[k].levels[0]||"(none)";
 var col=(d.values[k]||d.stats[k]).filter(function(v){return v!==null});return col.length?fmt(col[0]):"–"}
function layers(){var d=S.data,box=$("layerbody"),names=[].concat(LAYERS,["outcomes"]);
 names=names.filter(function(l){return settingsOf(l).length});
 if(names.indexOf(S.layer)<0)S.layer=names[0];
 $("layerpick").innerHTML=names.map(function(l){var n=settingsOf(l).filter(varies).length;
  return '<button type=button class="tog'+(l===S.layer?" on":"")+'" data-l="'+l+'">'+l.charAt(0).toUpperCase()+l.slice(1)+' <span class=mut>'+n+'</span></button>'}).join("");
 var keys=settingsOf(S.layer),vary=keys.filter(varies),still=keys.filter(function(k){return !varies(k)});
 var notes=d.notes||{};
 box.innerHTML=vary.map(function(k){var kind=k in d.factors?"f":S.layer==="outcomes"?"s":"v",note=notes[k]||"";
  return '<div class="card lpanel">'+(note?'<div class=cardinfo><span class=info tabindex=0 data-tip="'+esc(note)+'">ⓘ</span></div>':'')+'<h3 title="'+esc(k+(note?": "+note:""))+'">'+esc(plain(k))+'</h3><div data-'+kind+'="'+esc(k)+'"></div></div>'}).join("");
 $("constants").innerHTML=still.length?"<b>Constant in this source:</b> "+still.map(function(k){return esc(plain(k))+" "+esc(constantText(k))}).join("; "):"";
 var shown=S.shown;
 [].forEach.call(box.querySelectorAll("[data-f]"),function(el){var k=el.dataset.f,f=d.factors[k],
  na=new Array(f.levels.length).fill(0),ns=na.slice();
  f.codes.forEach(function(c){na[c]++});shown.forEach(function(i){ns[f.codes[i]]++});
  var sel=S.levels[k];
  barsH(el,f.levels.map(function(l,j){return {label:l===""?"(none)":l,level:l,n:ns[j],all:na[j],on:sel&&sel.has(l)}}),
   function(r){toggleLevel(k,r.level)})});
 [["v","values"],["s","stats"]].forEach(function(p){
  [].forEach.call(box.querySelectorAll("[data-"+p[0]+"]"),function(el){var k=el.dataset[p[0]],col=d[p[1]][k];
   hist(el,col,shown.map(function(i){return col[i]}),S.bins[k],function(r){
    var c=S.bins[k];if(c&&Math.abs(c[0]-r[0])<1e-12)delete S.bins[k];else S.bins[k]=r;update()})})})}
$("layerpick").addEventListener("click",function(e){var b=e.target.closest("button[data-l]");if(!b)return;S.layer=b.dataset.l;layers()});

// ---- what the simulations produce, by group -------------------------------
// Each summary computed from the simulations (the delay distribution,
// weekday profiles, release cadence, epidemic shape, revisions and noise),
// split by the "Colour and group by" factor as the series panels are: per
// level its median and 50% band, the source's whole faint grey.
var DW=300,DH=170,DAYS=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];
function q(a,p){var b=a.filter(function(v){return v!==null&&!isNaN(v)}).sort(function(x,y){return x-y});if(!b.length)return null;
 var i=(b.length-1)*p,lo=Math.floor(i),hi=Math.ceil(i);return b[lo]+(b[hi]-b[lo])*(i-lo)}
function dsvg(inner){return '<svg viewBox="0 0 '+DW+' '+DH+'" width="100%" role="img">'+inner+'</svg>'}
// curves: per group, the median and 50% band of each point of a curve.
function curvePanel(el,xs,xlab,rowsOf,groupsList,colour,opts){
 var L=36,R=8,T=8,B=28,pw=DW-L-R,ph=DH-T-B,g=[],ys=[];
 var sets=groupsList.map(function(gr){var rows=rowsOf(gr.ix);
  var st=xs.map(function(_,j){var col=rows.map(function(r){return r[j]});return [q(col,0.25),q(col,0.5),q(col,0.75)]});
  st.forEach(function(v){v.forEach(function(x){if(x!==null)ys.push(x)})});return {gr:gr,st:st}});
 if(!ys.length){el.innerHTML="<p class=mut>No values.</p>";return}
 var y0=opts.y0!==undefined?opts.y0:Math.min.apply(null,ys),y1=Math.max.apply(null,ys);if(y1===y0)y1=y0+1;
 var x0=0,x1=xs.length-1,sx=function(j){return L+(j-x0)/(x1-x0||1)*pw},sy=function(v){return T+ph-(v-y0)/(y1-y0)*ph};
 ticks(y0,y1,3).forEach(function(v){g.push('<line class=grid x1='+L+' x2='+(DW-R)+' y1='+sy(v)+' y2='+sy(v)+'></line><text class=ax x='+(L-4)+' y='+(sy(v)+3)+' text-anchor=end>'+fmt(v)+'</text>')});
 if(opts.hline!==undefined)g.push('<line class=mark x1='+L+' x2='+(DW-R)+' y1='+sy(opts.hline)+' y2='+sy(opts.hline)+'></line>');
 xs.forEach(function(x,j){if(xs.length<=8||j%2===0||j===xs.length-1)g.push('<text class=ax x='+sx(j)+' y='+(DH-B+12)+' text-anchor=middle>'+x+'</text>')});
 g.push('<text class=ax x='+(L+pw/2)+' y='+(DH-3)+' text-anchor=middle>'+xlab+'</text>');
 sets.forEach(function(set){var c=colour(set.gr.key),faint=set.gr.key==="(all)",up=[],dn=[],med="";
  set.st.forEach(function(v,j){if(v[0]===null)return;up.push(sx(j).toFixed(1)+","+sy(v[2]).toFixed(1));dn.unshift(sx(j).toFixed(1)+","+sy(v[0]).toFixed(1));med+=(med?"L":"M")+sx(j).toFixed(1)+" "+sy(v[1]).toFixed(1)});
  g.push('<polygon points="'+up.concat(dn).join(" ")+'" fill="'+c+'" opacity="'+(faint?0.12:0.18)+'"><title>'+esc(set.gr.label)+': 50% of simulations</title></polygon>');
  g.push('<path d="'+med+'" fill=none stroke="'+c+'" stroke-width='+(faint?1.5:2)+(faint?' stroke-dasharray="4 3"':'')+'><title>'+esc(set.gr.label)+' ('+set.gr.ix.length+'): median</title></path>')});
 el.innerHTML=dsvg(g.join(""))}
// intervals: per group a row, its 5-95% line, 25-75% bar and median.
function intervalPanel(el,col,groupsList,colour,log){
 var L=92,R=10,T=6,B=18,rh=Math.min(18,(DH-T-B)/groupsList.length),pw=DW-L-R,g=[],all=[];
 var sets=groupsList.map(function(gr){var v=gr.ix.map(function(i){return col[i]}).filter(function(x){return x!==null&&(!log||x>0)});v.forEach(function(x){all.push(x)});
  return {gr:gr,q:[q(v,0.05),q(v,0.25),q(v,0.5),q(v,0.75),q(v,0.95)],n:v.length}});
 if(!all.length){el.innerHTML="<p class=mut>No values.</p>";return}
 var tf=log?Math.log10:function(v){return v},inv=log?function(v){return Math.pow(10,v)}:function(v){return v};
 var lo=tf(q(all,0.01)),hi=tf(q(all,0.99));if(hi===lo){hi=lo+1}
 var sx=function(v){return L+(Math.min(hi,Math.max(lo,tf(v)))-lo)/(hi-lo)*pw};
 [lo,(lo+hi)/2,hi].forEach(function(t,k){g.push('<text class=ax x='+(L+k*pw/2)+' y='+(DH-4)+' text-anchor='+(k===0?'start':k===2?'end':'middle')+'>'+fmt(inv(t))+'</text>')});
 sets.forEach(function(set,i){var y=T+i*rh+rh/2,c=colour(set.gr.key),v=set.q;
  g.push('<text class=ax x='+(L-6)+' y='+(y+4)+' text-anchor=end>'+esc(set.gr.label.length>13?set.gr.label.slice(0,12)+"…":set.gr.label)+'</text>');
  if(v[2]===null)return;
  g.push('<g><title>'+esc(set.gr.label)+' ('+set.n+'): median '+fmt(v[2])+', 50% '+fmt(v[1])+' to '+fmt(v[3])+', 90% '+fmt(v[0])+' to '+fmt(v[4])+'</title>'+
   '<line x1='+sx(v[0])+' x2='+sx(v[4])+' y1='+y+' y2='+y+' stroke="'+c+'" stroke-width=1.5 opacity=.7></line>'+
   '<rect x='+sx(v[1])+' y='+(y-4)+' width='+Math.max(1,sx(v[3])-sx(v[1]))+' height=8 rx=2 fill="'+c+'" opacity=.45></rect>'+
   '<circle cx='+sx(v[2])+' cy='+y+' r=3.5 fill="'+c+'" stroke="var(--card)" stroke-width=1.5></circle>'+
   '<rect x=0 y='+(y-rh/2)+' width='+DW+' height='+rh+' fill=transparent></rect></g>')});
 if(log)g.push('<text class=ax x='+(DW-R)+' y='+(T+8)+' text-anchor=end>log scale</text>');
 el.innerHTML=dsvg(g.join(""))}
function derived(){var d=S.data,box=$("derivedbody");if(!d.derived){box.innerHTML='<p class=mut>This source was summarised before these summaries; it is summarised again on the next refresh.</p>';return}
 var g=groups(),by={};S.shown.forEach(function(i){var l=g.of(i);(by[l]=by[l]||[]).push(i)});
 var list=[{key:"(all)",label:"all of the source",ix:d.ids.map(function(_,i){return i})}].concat(g.list.map(function(l){return {key:l,label:l==="all"?"shown":l,ix:by[l]||[]}}));
 var colour=function(k){return k==="(all)"?"var(--mut)":g.colour(k)};
 $("derivedleg").innerHTML=list.map(function(x){return '<span><i style="background:'+colour(x.key)+(x.key==="(all)"?';opacity:.5':'')+'"></i>'+esc(x.label)+' <span class=mut>'+x.ix.length+'</span></span>'}).join("")+'<span class=mut>Split by '+esc(S.group||"nothing")+' (the "Colour and group by" control); median and 50% band, or median, 50% and 90% of the simulations.</span>';
 var curves=[["cdf","Share reported by delay","delay (days)",d.delay_points,{y0:0}],["refdow","Reference-date weekday profile","weekday of reference date",DAYS,{hline:1}],["repdow","Report-date weekday profile","weekday of report",DAYS,{hline:1}]];
 var scalars=[["within7"],["within14"],["within28"],["releases"],["gap"],["first_report"],["growth"],["peak",true],["peak_at"],["negative"],["noise"]];
 box.innerHTML=curves.map(function(c){return '<div class="card lpanel"><h3>'+c[1]+'</h3><div data-c="'+c[0]+'"></div></div>'}).join("")+
  scalars.filter(function(k){return d.stats[k[0]]}).map(function(k){return '<div class="card lpanel"><h3>'+esc(STATS[k[0]]||k[0])+'</h3><div data-x="'+k[0]+'"></div></div>'}).join("");
 curves.forEach(function(c){curvePanel(box.querySelector('[data-c="'+c[0]+'"]'),c[3],c[2],function(ix){return ix.map(function(i){return d.derived[c[0]][i]})},list,colour,c[4])});
 scalars.forEach(function(k){var el=box.querySelector('[data-x="'+k[0]+'"]');if(el)intervalPanel(el,d.stats[k[0]],list,colour,!!k[1])})}

function sample(ix,cap){if(ix.length<=cap)return ix;var out=[],step=ix.length/cap;
 for(var j=0;j<cap;j++)out.push(ix[Math.floor(j*step)]);return out}
function seriesOf(i,colour,op,w){var y=S.data.series[i],pk=Math.max.apply(null,y)||1;
 return {x:y.map(function(_,j){return j}),y:S.norm?y.map(function(v){return v/pk}):y,
  colour:colour,op:op,w:w||1,id:S.data.ids[i],name:S.data.ids[i]}}
function spaghetti(){var g=groups(),cap=Math.max(10,+$("cap").value||150);
 var hover=function(s,j){var i=S.data.ids.indexOf(s.id);return "<b>"+esc(s.id)+"</b>"+(S.group?" ("+esc(level(S.group,i))+")":"")+
  "<br>week "+j+": "+fmt(s.y[j])+"<br>"+fmt(S.data.stats.cases[i])+" cases, "+S.data.stats.strata[i]+" strata"};
 var opts={log:S.scale==="log",xlab:"Weeks since first reference date",ylab:S.norm?"Share of peak week":"Weekly cases",pick:openSim,hover:hover};
 var op=function(n){return Math.max(0.08,Math.min(0.8,3/Math.sqrt(n+1)))};
 var pick=sample(S.shown,cap*2);
 lineChart($("spag"),Object.assign({series:pick.map(function(i){return seriesOf(i,g.colour(g.of(i)),op(pick.length))}),height:300},opts));
 $("spagleg").innerHTML=g.list.length>1?g.list.map(function(l){return '<span><i style="background:'+g.colour(l)+'"></i>'+esc(l)+"</span>"}).join(""):"";
 var by={};S.shown.forEach(function(i){var l=g.of(i);(by[l]=by[l]||[]).push(i)});
 var box=$("panels");
 if(g.list.length<2){box.innerHTML="";return}
 box.innerHTML=g.list.map(function(l,j){return '<div class=card><h3><i class=sw style="background:'+g.colour(l)+'"></i>'+esc(l)+' <span class=mut>('+(by[l]||[]).length+')</span></h3><div data-p='+j+'></div></div>'}).join("");
 g.list.forEach(function(l,j){var ix=sample(by[l]||[],cap);
  lineChart(box.querySelector("[data-p='"+j+"']"),Object.assign({series:ix.map(function(i){return seriesOf(i,g.colour(l),op(ix.length))}),height:200},opts))})}

function table(){var d=S.data,cols=["cases","strata","weeks","peak","peak_at","delay","within7"],key=S.sort;
 var ix=S.shown.slice().sort(function(a,b){var x=d.stats[key][a],y=d.stats[key][b];return S.desc?y-x:x-y}).slice(0,200);
 var fac=S.group?[S.group]:[];
 var h='<tr><th>Simulation</th>'+fac.map(function(f){return "<th>"+esc(f)+"</th>"}).join("")+
  cols.map(function(c){return '<th data-c='+c+'>'+STATS[c]+(c===key?(S.desc?" ▾":" ▴"):"")+"</th>"}).join("")+"</tr>";
 ix.forEach(function(i){h+='<tr data-id="'+d.ids[i]+'"><td>'+d.ids[i]+"</td>"+fac.map(function(f){return "<td>"+esc(level(f,i))+"</td>"}).join("")+
  cols.map(function(c){return "<td>"+fmt(d.stats[c][i])+"</td>"}).join("")+"</tr>"});
 $("simtable").innerHTML=h;
 [].forEach.call($("simtable").querySelectorAll("th[data-c]"),function(th){th.addEventListener("click",function(){
  if(S.sort===th.dataset.c)S.desc=!S.desc;else{S.sort=th.dataset.c;S.desc=true}table()})});
 [].forEach.call($("simtable").querySelectorAll("tr[data-id]"),function(tr){tr.addEventListener("click",function(){openSim(tr.dataset.id)})})}

// ---- deep dive ------------------------------------------------------------
function dayNum(s){return Date.parse(s+"T00:00:00Z")/864e5}
function dayLabel(n){return new Date(n*864e5).toISOString().slice(0,10)}
function openSim(id){S.sim=id;$("simid").value=id;setHash();$("dive").open=true;
 $("divestatus").textContent="loading…";
 fetch("sim/"+encodeURIComponent(S.src)+"/"+encodeURIComponent(id)+".json").then(function(r){
  if(!r.ok)throw new Error(r.status===404?"no such simulation in "+S.src:"error "+r.status);return r.json()})
  .then(function(j){$("divestatus").textContent="";dive(j);$("dive").scrollIntoView({behavior:"smooth"})})
  .catch(function(e){$("divestatus").textContent=e.message;$("divebody").innerHTML=""})}
function dive(j){var box=$("divebody"),x=j.dates.map(dayNum),i=S.data.ids.indexOf(j.id);
 var cards=[["strata","Final counts by stratum"],["known","As known at three cut-offs"],["delays","Share reported by delay"],
  ["tri","Reporting triangle (weeks)"]];
 if(j.latent)cards.push(["lat","Infections and onsets"],["r","R and effective R"]);
 var head=i>=0?'<p class=cnote>'+fmt(S.data.stats.cases[i])+" cases over "+S.data.stats.days[i]+" days in "+S.data.stats.strata[i]+
  " strata; mean delay "+fmt(S.data.stats.delay[i])+" days. "+LAYERS.map(function(l){var fs=Object.keys(S.data.factors).filter(function(k){return S.data.layers[k]===l}).slice(0,4);
   return fs.length?"<b>"+l+"</b>: "+fs.map(function(k){return esc(level(k,i))}).join(", "):""}).filter(Boolean).join("; ")+".</p>":"";
 box.innerHTML='<h3>'+esc(j.id)+'</h3>'+head+'<div class=plots>'+cards.map(function(c){return '<div class=card><h3>'+c[1]+'</h3><div id=d-'+c[0]+'></div><div id=l-'+c[0]+' class=leg></div></div>'}).join("")+
  '</div><h3 style="margin-top:10px">Settings by layer</h3><div class=plots3 id=d-set></div>';
 // Strata: the eight largest each, the rest summed as "other", and the total.
 var names=Object.keys(j.strata),tot=function(n){return j.strata[n].reduce(function(a,b){return a+b},0)};
 names.sort(function(a,b){return tot(b)-tot(a)});
 var keep=names.slice(0,names.length>PAL.length?MAXCOL:PAL.length),rest=names.slice(keep.length),ser=[];
 keep.forEach(function(n,k){ser.push({x:x,y:j.strata[n],colour:PAL[k],name:n,w:1.5})});
 if(rest.length)ser.push({x:x,y:x.map(function(_,t){return rest.reduce(function(a,n){return a+j.strata[n][t]},0)}),colour:"var(--mut)",name:"other ("+rest.length+")",w:1.5});
 var total=x.map(function(_,t){return names.reduce(function(a,n){return a+j.strata[n][t]},0)});
 if(names.length>1)ser.push({x:x,y:total,colour:"var(--fg)",name:"total",w:2});
 lineChart($("d-strata"),{series:ser,dates:dayLabel,ylab:"Cases",hover:defaultHover});
 $("l-strata").innerHTML=ser.length>1?ser.map(function(s){return '<span><i style="background:'+s.colour+'"></i>'+esc(s.name)+"</span>"}).join(""):"";
 var kn=j.known.map(function(k,q){return {x:x.slice(0,k.counts.length),y:k.counts,colour:PAL[q+1],name:"known at "+k.at,w:2}});
 kn.unshift({x:x,y:total,colour:"var(--fg)",name:"final",w:2});
 lineChart($("d-known"),{series:kn,dates:dayLabel,ylab:"Cases",hover:defaultHover});
 $("l-known").innerHTML=kn.map(function(s){return '<span><i style="background:'+s.colour+'"></i>'+esc(s.name)+"</span>"}).join("");
 var cum=0,cy=j.delays.map(function(v){cum+=v;return cum});
 lineChart($("d-delays"),{series:[{x:j.delays.map(function(_,k){return k}),y:cy,colour:PAL[0],name:"cumulative share"},
  {x:j.delays.map(function(_,k){return k}),y:j.delays,colour:PAL[1],name:"share at delay",w:1.5}],xlab:"Delay (days)",ylab:"Share of cases",hover:defaultHover,ymin:0});
 $("l-delays").innerHTML='<span><i style="background:'+PAL[0]+'"></i>cumulative share</span><span><i style="background:'+PAL[1]+'"></i>share at each delay</span>';
 heat($("d-tri"),j.triangle.counts,"delay (weeks)","reference week");
 $("l-tri").innerHTML='<span class=mut>Rows: weeks of reference date; columns: weeks of delay; darker is more cases (log).</span>';
 if(j.latent){var lx=j.latent.dates.map(dayNum);
  lineChart($("d-lat"),{series:[{x:lx,y:j.latent.infections,colour:PAL[0],name:"infections"},{x:lx,y:j.latent.onsets,colour:PAL[1],name:"expected onsets"}],dates:dayLabel,hover:defaultHover});
  $("l-lat").innerHTML='<span><i style="background:'+PAL[0]+'"></i>infections</span><span><i style="background:'+PAL[1]+'"></i>expected onsets</span>';
  lineChart($("d-r"),{series:[{x:lx,y:j.latent.r,colour:PAL[0],name:"R"},{x:lx,y:j.latent.reff,colour:PAL[1],name:"effective R"}],dates:dayLabel,hline:1,hover:defaultHover});
  $("l-r").innerHTML='<span><i style="background:'+PAL[0]+'"></i>R</span><span><i style="background:'+PAL[1]+'"></i>effective R</span>'}
 var by={};j.settings.forEach(function(s){(by[s.layer]=by[s.layer]||[]).push(s)});
 $("d-set").innerHTML=[].concat(LAYERS,["other"]).filter(function(l){return by[l]}).map(function(l){
  return '<div class=card><h3 style="text-transform:capitalize">'+l+'</h3><table class=settings>'+by[l].map(function(s){
   // A value per stratum, lag or day: its first six, the whole on hover.
   var parts=s.value.split(";"),one=function(p){var n=parseFloat(p);return isNaN(n)||/[a-z]/i.test(p)?p:fmt(n)};
   var v=parts.map(one).slice(0,6).join("; ")+(parts.length>6?" … ("+parts.length+")":"");
   return "<tr><td>"+esc(s.name)+'</td><td title="'+esc(parts.length>6?parts.map(one).join("; "):"")+'">'+esc(v)+"</td></tr>"}).join("")+"</table></div>"}).join("")}

// ---- wiring ------------------------------------------------------------
function update(){S.shown=filtered();chips();
 $("status").textContent=S.shown.length.toLocaleString()+" of "+S.data.ids.length.toLocaleString()+" shown";
 summary();layers();derived();spaghetti();table()}
function setHash(){var h="src="+encodeURIComponent(S.src||"")+(S.sim?"&sim="+encodeURIComponent(S.sim):"");
 history.replaceState(null,"","#"+h)}
function readHash(){var o={};location.hash.slice(1).split("&").forEach(function(p){var kv=p.split("=");if(kv[0])o[kv[0]]=decodeURIComponent(kv[1]||"")});return o}
function load(src,sim){S.src=src;S.levels={};S.bins={};S.sim=sim||null;setHash();
 $("status").textContent="loading "+src+"… (a source is summarised once, then cached)";
 fetch("index/"+encodeURIComponent(src)+".json").then(function(r){if(!r.ok)throw new Error("no index for "+src);return r.json()}).then(function(d){
  S.data=d;var fs=factorNames();
  $("groupby").innerHTML='<option value="">(none)</option>'+LAYERS.map(function(l){
   var k=fs.filter(function(f){return d.layers[f]===l});return k.length?'<optgroup label="'+l+'">'+k.map(function(f){return "<option>"+esc(f)+"</option>"}).join("")+"</optgroup>":""}).join("");
  if(!S.group||fs.indexOf(S.group)<0)S.group=fs.indexOf("trajectory")>=0?"trajectory":null;
  $("groupby").value=S.group||"";
  $("simids").innerHTML=d.ids.slice(0,5000).map(function(i){return "<option value='"+i+"'>"}).join("");
  update();if(S.sim)openSim(S.sim)}).catch(function(e){$("status").textContent=e.message})}
$("groupby").addEventListener("change",function(){S.group=this.value||null;update()});
$("clear").addEventListener("click",function(){S.levels={};S.bins={};update()});
$("source").addEventListener("change",function(){load(this.value)});
$("cap").addEventListener("change",spaghetti);
$("norm").addEventListener("click",function(){S.norm=!S.norm;this.classList.toggle("on",S.norm);spaghetti()});
[].forEach.call(document.querySelectorAll("[data-scale]"),function(b){b.addEventListener("click",function(){
 S.scale=b.dataset.scale;[].forEach.call(document.querySelectorAll("[data-scale]"),function(o){o.classList.toggle("on",o===b)});spaghetti()})});
$("simid").addEventListener("change",function(){if(this.value)openSim(this.value.trim())});
$("random").addEventListener("click",function(){if(S.shown.length)openSim(S.data.ids[S.shown[Math.floor(Math.random()*S.shown.length)]])});
window.addEventListener("hashchange",function(){var h=readHash();
 if(h.src&&h.src!==S.src){$("source").value=h.src;load(h.src,h.sim)}
 else if(h.sim&&h.sim!==S.sim)openSim(h.sim)});
var resized;window.addEventListener("resize",function(){clearTimeout(resized);resized=setTimeout(function(){if(S.data)update()},250)});
fetch("sources.json").then(function(r){return r.json()}).then(function(list){
 if(!list.length){$("status").textContent="No simulations under this machine's ROOT yet.";return}
 $("source").innerHTML=list.map(function(s){return "<option value='"+esc(s.key)+"'>"+esc(s.key)+(s.simulations!==null?" ("+s.simulations+")":"")+"</option>"}).join("");
 var h=readHash(),keys=list.map(function(s){return s.key}),def=list.find(function(s){return s.default}),
  src=keys.indexOf(h.src)>=0?h.src:def?def.key:keys[keys.length-1];
 $("source").value=src;load(src,h.sim)}).catch(function(){$("status").textContent="Could not reach the server."});
})();
