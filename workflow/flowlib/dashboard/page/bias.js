// The attention bias section: one even grid of equal panels (PW x PH, as
// page.BIAS_PANEL and the learning rate's card), by relation in the
// trainer's order. The first row is each relation's bias over training
// (mean, min to max over layers and heads; every selected run, its line
// style) then the learning rate; the second is each relation's latest
// layer x head matrix then the mean by layer. Scales are shared along a
// row; one legend and colour key sit above the grid.
var PW=320,PH=180;
// biasLayout: the grid's columns and its panels in order, [kind, relation].
function biasLayout(pairs){var n=pairs.length,out=[];
 pairs.forEach(function(k){out.push(["line",k.key])});out.push(["lr",null]);
 pairs.forEach(function(k){out.push(["heat",k.key])});out.push(["layers",null]);
 return {cols:n+1,panels:out}}
if(typeof module!=="undefined")module.exports={biasLayout:biasLayout};
else (function(){
var D=JSON.parse(document.getElementById("bdata").textContent);
var $=function(i){return document.getElementById(i)};
var sel0=null,shape="";
var STY=["","6 3","2 3","8 3 2 3"];
function fmt(v){return String(+(+v).toPrecision(3))}
function nice(lo,hi,n){var st=(hi-lo)/(n||4),m=Math.pow(10,Math.floor(Math.log10(st))),k=[1,2,2.5,5,10].find(function(x){return x*m>=st})*m,o=[];
 for(var v=Math.ceil(lo/k)*k;v<=hi+1e-9;v+=k)o.push(+v.toFixed(6));return o}
function sw(c,d){return '<svg width=22 height=8 viewBox="0 0 22 8"><line x1=0 x2=22 y1=4 y2=4 stroke="'+c+'" stroke-width=2'+(d?' stroke-dasharray="'+d+'"':"")+' /></svg>'}
function svgOpen(){return '<svg viewBox="0 0 '+PW+' '+PH+'" width="100%" role="img">'}
// Diverging fill: the second accent above 0, the first below, faint near 0.
function fill(v,amp){var a=Math.min(1,Math.abs(v)/amp);
 return 'fill="var('+(v>0?"--c2":"--c1")+')" fill-opacity="'+(0.06+0.94*a).toFixed(2)+'"'}
function frame(L,R,T,B,lo,hi,x0,x1,xlab){var sy=function(y){return T+(1-(y-lo)/(hi-lo))*(PH-T-B)},sx=function(x){return L+(x-x0)/((x1-x0)||1)*(PW-L-R)},s="";
 nice(lo,hi,3).forEach(function(v){if(v>=lo&&v<=hi)s+='<line x1="'+L+'" x2="'+(PW-R)+'" y1="'+sy(v).toFixed(1)+'" y2="'+sy(v).toFixed(1)+'" class=grid /><text x="'+(L-4)+'" y="'+(sy(v)+3).toFixed(1)+'" text-anchor=end class=ax>'+fmt(v)+'</text>'});
 nice(x0,x1,3).forEach(function(v){if(v>=x0&&v<=x1)s+='<text x="'+sx(v).toFixed(1)+'" y="'+(PH-B+13)+'" text-anchor=middle class=ax>'+v+'</text>'});
 s+='<text x="'+((L+PW-R)/2)+'" y="'+(PH-3)+'" text-anchor=middle class=ax>'+xlab+'</text>';
 if(lo<0&&hi>0)s+='<line x1="'+L+'" x2="'+(PW-R)+'" y1="'+sy(0).toFixed(1)+'" y2="'+sy(0).toFixed(1)+'" class=mark />';
 return {sx:sx,sy:sy,s:s}}
function line(runs,k,lim){var f=frame(40,8,6,30,lim.lo,lim.hi,0,lim.x1,"simulations trained"),s=svgOpen()+f.s,c=D.col[k.key];
 runs.forEach(function(run,ri){var pts=run.pts[k.key]||[];if(!pts.length)return;var d=STY[ri%STY.length];
  var up=pts.map(function(p){return f.sx(p[0]).toFixed(1)+","+f.sy(p[3]).toFixed(1)}).join(" "),dn=pts.slice().reverse().map(function(p){return f.sx(p[0]).toFixed(1)+","+f.sy(p[2]).toFixed(1)}).join(" "),mean=pts.map(function(p){return f.sx(p[0]).toFixed(1)+","+f.sy(p[1]).toFixed(1)}).join(" ");
  s+='<polygon points="'+up+" "+dn+'" fill="'+c+'" fill-opacity=.15 /><polyline points="'+mean+'" fill=none stroke="'+c+'" stroke-width=2'+(d?' stroke-dasharray="'+d+'"':"")+' />';
  var last=pts[pts.length-1];
  s+='<circle cx="'+f.sx(last[0]).toFixed(1)+'" cy="'+f.sy(last[1]).toFixed(1)+'" r=6 fill=transparent><title>'+run.label+', '+k.label+' at '+Math.round(last[0])+' simulations: mean '+fmt(last[1])+', range '+fmt(last[2])+' to '+fmt(last[3])+'</title></circle>'});
 return s+'</svg>'}
function heat(rows,k,amp){var nr=rows.length,nc=rows[0].length,L=40,T=6,B=30,side=Math.min(PH-T-B,PW-L-8),cw=side/Math.max(nr,nc),x0=L+(PW-L-8-nc*cw)/2,s=svgOpen();
 rows.forEach(function(r,i){r.forEach(function(v,j){s+='<rect x="'+(x0+j*cw).toFixed(1)+'" y="'+(T+i*cw).toFixed(1)+'" width="'+(cw-0.6).toFixed(1)+'" height="'+(cw-0.6).toFixed(1)+'" '+fill(v,amp)+'><title>'+k.label+', layer '+i+' head '+j+': '+fmt(v)+'</title></rect>'})});
 [0,nr-1].forEach(function(i){s+='<text x="'+(x0-4)+'" y="'+(T+i*cw+cw*0.8).toFixed(1)+'" text-anchor=end class=ax>'+i+'</text>'});
 [0,nc-1].forEach(function(j){s+='<text x="'+(x0+j*cw+cw/2).toFixed(1)+'" y="'+(T+nr*cw+12).toFixed(1)+'" text-anchor=middle class=ax>'+j+'</text>'});
 s+='<text x="'+(x0-4)+'" y="'+(T+nr*cw/2+3)+'" text-anchor=end class=ax>layer</text><text x="'+(x0+nc*cw/2)+'" y="'+(PH-3)+'" text-anchor=middle class=ax>head</text>';
 return s+'</svg>'}
function layers(lat,keys,lim){var n=0;keys.forEach(function(k){n=Math.max(n,lat[k.key].length)});
 var f=frame(40,8,6,30,lim.lo,lim.hi,0,Math.max(n-1,1),"layer"),s=svgOpen()+f.s;
 keys.forEach(function(k){var m=lat[k.key].map(function(r){return r.reduce(function(a,b){return a+b},0)/r.length}),c=D.col[k.key];
  s+='<polyline points="'+m.map(function(v,i){return f.sx(i).toFixed(1)+","+f.sy(v).toFixed(1)}).join(" ")+'" fill=none stroke="'+c+'" stroke-width=2 />';
  m.forEach(function(v,i){s+='<circle cx="'+f.sx(i).toFixed(1)+'" cy="'+f.sy(v).toFixed(1)+'" r=2.5 fill="'+c+'"><title>'+k.label+', layer '+i+': mean '+fmt(v)+'</title></circle>'})});
 return s+'</svg>'}
function card(id,title,sub,info){return '<div class="card bpanel" id="'+id+'"><h3>'+title+(info?ftInfo(info):"")+'</h3><div class="leg bsub">'+sub+'</div><div class=bplot></div></div>'}
function draw(){
 var on=ftSel.get(),runs=D.runs.filter(function(r){return on.indexOf(r.key)>=0}),grid=$("biasgrid"),head=$("biashead"),lr=$("lrcard");
 $("biasrefs").innerHTML=ftSel.refsOn().map(function(r){return '<span class="tog dis" title="'+r.label+' is a reference with no attention bias">'+r.label+'</span>'}).join("");
 if(lr)lr.style.display=on.indexOf(lr.dataset.run)>=0?"":"none";
 var none=(D.missing||[]).filter(function(r){return on.indexOf(r.key)>=0}).map(function(r){return r.label});
 var miss=none.length?'<span class=mut>no attention bias logged or saved: '+none.join(", ")+'</span>':"";
 if(!runs.length){head.innerHTML='<p class=mut>none of the selected runs has an attention bias module</p>';
  [].forEach.call(grid.querySelectorAll(".bpanel"),function(c){c.remove()});shape="";return}
 if(!sel0||!runs.some(function(r){return r.key==sel0}))sel0=runs[0].key;
 var run=runs.find(function(r){return r.key==sel0}),lat=run.latest||{};
 var pairs=D.pairs.filter(function(k){return runs.some(function(r){return (r.pts[k.key]||[]).length})||lat[k.key]});
 var lay=biasLayout(pairs),keys=pairs.filter(function(k){return lat[k.key]});
 // Shared scales: one for the bias over training, one for the matrices.
 var ys=[],x1=1,flat=[];
 runs.forEach(function(r){pairs.forEach(function(k){(r.pts[k.key]||[]).forEach(function(p){ys.push(p[2],p[3]);x1=Math.max(x1,p[0])})})});
 keys.forEach(function(k){lat[k.key].forEach(function(r){r.forEach(function(v){flat.push(v)})})});
 var lo=ys.length?Math.min.apply(null,ys):-1,hi=ys.length?Math.max.apply(null,ys):1,pad=(hi-lo)*0.08||0.1;
 var lim={lo:Math.min(lo-pad,0),hi:Math.max(hi+pad,0),x1:x1};
 var amp=flat.length?Math.max.apply(null,flat.map(Math.abs))||1:1;
 var means=[];keys.forEach(function(k){lat[k.key].forEach(function(r){means.push(r.reduce(function(a,b){return a+b},0)/r.length)})});
 var mlo=means.length?Math.min.apply(null,means):-1,mhi=means.length?Math.max.apply(null,means):1,mp=(mhi-mlo)*0.1||0.1;
 // The legend: relations, runs' line styles, the matrices' run and key.
 var tabs=runs.length>1?'<span class=lg><span class=mut>matrices of</span>'+runs.map(function(r){return '<button class="tog'+(r.key==sel0?" on":"")+'" data-b="'+r.key+'" style="border-left:4px solid '+r.color+'">'+r.label+'</button>'}).join("")+'</span>':"";
 var key='<span class=lg><span class=mut>matrix</span><svg width=120 height=10 viewBox="0 0 120 10">'+[0,1,2,3,4,5,6,7,8,9,10,11].map(function(i){var v=(i-5.5)/5.5*amp;return '<rect x="'+(i*10)+'" y=0 width=9.4 height=10 '+fill(v,amp)+' />'}).join("")+'</svg><span class=mut>'+fmt(-amp)+' to '+fmt(amp)+'</span></span>';
 head.innerHTML='<div class=leg2>'+pairs.map(function(k){return '<span><i class=sw2 style="background:'+D.col[k.key]+'"></i>'+k.label+'</span>'}).join("")+
  (runs.length>1?runs.map(function(r,ri){return '<span>'+sw("var(--fg)",STY[ri%STY.length])+' '+r.label+'</span>'}).join(""):"")+
  (keys.length?key:"")+tabs+miss+'</div>';
 var tb=head.querySelector(".lg button")?head:null;if(tb)tb.onclick=function(e){var b=e.target.closest("button[data-b]");if(b){sel0=b.dataset.b;draw()}};
 // The panels are built once per layout and redrawn in place, so a card
 // open full screen stays open (expand.js).
 var sig=lay.panels.map(function(p){return p.join(":")}).join(",");
 grid.style.setProperty("--bcols",lay.cols);
 if(sig!==shape){shape=sig;[].forEach.call(grid.querySelectorAll(".bpanel"),function(c){c.remove()});
  var t=document.createElement("template");
  t.innerHTML=lay.panels.map(function(p){var k=D.pairs.find(function(x){return x.key===p[1]})||{};
   if(p[0]==="line")return card("bl-"+p[1],k.label,"mean, min to max over layers and heads",p[1]===pairs[0].key?"biasline":"");
   if(p[0]==="heat")return card("bh-"+p[1],k.label,"latest, by layer and head",p[1]===pairs[0].key?"biasheat":"");
   if(p[0]==="layers")return card("bm","Mean by layer","latest, each relation","relations");
   return '<div id=lrslot></div>'}).join("");
  grid.insertBefore(t.content,grid.firstChild);
  var slot=$("lrslot");if(slot){if(lr)slot.replaceWith(lr);else slot.remove()}}
 pairs.forEach(function(k){var c=$("bl-"+k.key);if(c)c.querySelector(".bplot").innerHTML=line(runs,k,lim);
  var h=$("bh-"+k.key);if(h)h.querySelector(".bplot").innerHTML=lat[k.key]?heat(lat[k.key],k,amp):'<p class=mut>no matrix saved</p>';
  if(h)h.querySelector(".bsub").textContent=lat[k.key]?run.label+" at "+Math.round(lat.n)+" simulations":"no matrix saved"});
 var m=$("bm");if(m)m.querySelector(".bplot").innerHTML=keys.length?layers(lat,keys,{lo:Math.min(mlo-mp,0),hi:Math.max(mhi+mp,0)}):'<p class=mut>no matrix saved</p>'}
window.addEventListener("ftsel",draw);
// New data from the server (live.js): redraw with the same state.
window.addEventListener("ftdata",function(e){if(e.detail!=="bias")return;D=JSON.parse($("bdata").textContent);draw()});
draw();
})();
