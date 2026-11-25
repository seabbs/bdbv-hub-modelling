// The attention bias section: one even grid of equal panels (PW x PH, as
// page.BIAS_PANEL and the learning rate's card), by relation in the
// trainer's order. The first row is each relation's bias over training
// (each layer's mean over heads for the run picked in the tabs, shallow
// light to deep dark in the relation's hue; every other selected run as
// its mean over layers and heads, its line style) then the learning
// rate; the second is each relation's latest layer x head matrix then
// the mean by layer. Scales are shared along a row; one legend and the
// keys sit above the grid.
var PW=320,PH=180;
// biasLayout: the grid's columns and its panels in order, [kind, relation].
function biasLayout(pairs){var n=pairs.length,out=[];
 pairs.forEach(function(k){out.push(["line",k.key])});out.push(["lr",null]);
 pairs.forEach(function(k){out.push(["heat",k.key])});out.push(["layers",null]);
 return {cols:n+1,panels:out}}
// layerShade: layer i of n in colour c, a single-hue ramp by depth. The
// shallowest is faint (c mixed into the card), the middle c itself and
// the deepest c mixed towards the ink, so depth reads in either theme.
function layerShade(c,i,n){if(n<2)return c;var t=i/(n-1);
 if(t<=0.6)return 'color-mix(in oklab, '+c+' '+Math.round(30+70*t/0.6)+'%, var(--card))';
 return 'color-mix(in oklab, '+c+' '+Math.round(100-45*(t-0.6)/0.4)+'%, var(--fg))'}
if(typeof module!=="undefined")module.exports={biasLayout:biasLayout,layerShade:layerShade};
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
// line: a relation's bias over training. The picked run's layers, each
// its mean over heads (or its overall mean if it saved no matrix), then
// the other runs' means. Returns the svg and its hover points.
function line(runs,run,k,lim){var f=frame(40,8,6,30,lim.lo,lim.hi,0,lim.x1,"simulations trained"),s=svgOpen()+f.s,c=D.col[k.key],hits=[];
 function draw(pts,col,d,label){if(!pts.length)return;
  s+='<polyline points="'+pts.map(function(p){return f.sx(p[0]).toFixed(1)+","+f.sy(p[1]).toFixed(1)}).join(" ")+'" fill=none style="stroke:'+col+'" stroke-width=1.5 stroke-linejoin=round'+(d?' stroke-dasharray="'+d+'"':"")+'><title>'+label+'</title></polyline>';
  pts.forEach(function(p){hits.push({x:f.sx(p[0]),y:f.sy(p[1]),c:col,t:label+' at '+Math.round(p[0])+' simulations: '+fmt(p[1])})})}
 var others=runs.filter(function(r){return r!==run});
 others.forEach(function(r,ri){draw((r.pts[k.key]||[]).map(function(p){return [p[0],p[1]]}),"var(--mut)",STY[1+ri%(STY.length-1)],r.label+', mean')});
 var ls=(run.layers||{})[k.key]||[],n=0;ls.forEach(function(p){n=Math.max(n,p[1].length)});
 if(n)for(var i=0;i<n;i++)draw(ls.filter(function(p){return p[1].length>i}).map(function(p){return [p[0],p[1][i]]}),layerShade(c,i,n),"",(runs.length>1?run.label+', ':'')+'layer '+i);
 else draw((run.pts[k.key]||[]).map(function(p){return [p[0],p[1]]}),c,"",run.label+', mean');
 return {svg:s+'<circle class=bhit r=4 cx=-9 cy=-9 stroke="var(--card)" stroke-width=2 style="display:none" /></svg>',hits:hits}}
// hover: on a line panel, the point nearest the pointer is ringed and
// named in the panel's subtitle (layer, simulations, value).
function hover(card,hits){var svg=card.querySelector("svg"),sub=card.querySelector(".bsub"),dot=svg.querySelector(".bhit");
 svg.onmousemove=function(e){var r=svg.getBoundingClientRect(),mx=(e.clientX-r.left)*PW/r.width,my=(e.clientY-r.top)*PH/r.height,best=null,bd=Infinity;
  hits.forEach(function(h){var d=Math.pow(h.x-mx,2)+Math.pow(h.y-my,2);if(d<bd){bd=d;best=h}});
  if(!best||bd>900){svg.onmouseleave();return}
  dot.setAttribute("cx",best.x.toFixed(1));dot.setAttribute("cy",best.y.toFixed(1));dot.style.fill=best.c;dot.style.display="";sub.textContent=best.t};
 svg.onmouseleave=function(){dot.style.display="none";sub.textContent=sub.dataset.d}}
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
 runs.forEach(function(r){pairs.forEach(function(k){(r.pts[k.key]||[]).forEach(function(p){ys.push(p[1]);x1=Math.max(x1,p[0])})})});
 pairs.forEach(function(k){((run.layers||{})[k.key]||[]).forEach(function(p){p[1].forEach(function(v){ys.push(v)});x1=Math.max(x1,p[0])})});
 var nl=0;pairs.forEach(function(k){((run.layers||{})[k.key]||[]).forEach(function(p){nl=Math.max(nl,p[1].length)})});
 keys.forEach(function(k){lat[k.key].forEach(function(r){r.forEach(function(v){flat.push(v)})})});
 var lo=ys.length?Math.min.apply(null,ys):-1,hi=ys.length?Math.max.apply(null,ys):1,pad=(hi-lo)*0.08||0.1;
 var lim={lo:Math.min(lo-pad,0),hi:Math.max(hi+pad,0),x1:x1};
 var amp=flat.length?Math.max.apply(null,flat.map(Math.abs))||1:1;
 var means=[];keys.forEach(function(k){lat[k.key].forEach(function(r){means.push(r.reduce(function(a,b){return a+b},0)/r.length)})});
 var mlo=means.length?Math.min.apply(null,means):-1,mhi=means.length?Math.max.apply(null,means):1,mp=(mhi-mlo)*0.1||0.1;
 // The legend: relations, runs' line styles, the matrices' run and key.
 var tabs=runs.length>1?'<span class=lg><span class=mut>layers and matrices of</span>'+runs.map(function(r){return '<button class="tog'+(r.key==sel0?" on":"")+'" data-b="'+r.key+'" style="border-left:4px solid '+r.color+'">'+r.label+'</button>'}).join("")+'</span>':"";
 var key='<span class=lg><span class=mut>matrix</span><svg width=120 height=10 viewBox="0 0 120 10">'+[0,1,2,3,4,5,6,7,8,9,10,11].map(function(i){var v=(i-5.5)/5.5*amp;return '<rect x="'+(i*10)+'" y=0 width=9.4 height=10 '+fill(v,amp)+' />'}).join("")+'</svg><span class=mut>'+fmt(-amp)+' to '+fmt(amp)+'</span></span>';
 head.innerHTML='<div class=leg2>'+pairs.map(function(k){return '<span><i class=sw2 style="background:'+D.col[k.key]+'"></i>'+k.label+'</span>'}).join("")+
  (nl>1?'<span class=lg><span class=mut>layer 0</span><svg width=72 height=8 viewBox="0 0 72 8">'+[0,1,2,3,4,5].map(function(i){return '<rect x="'+(i*12)+'" y=0 width=11 height=8 style="fill:'+layerShade("var(--fg)",i,6)+'" />'}).join("")+'</svg><span class=mut>'+(nl-1)+' (shallow to deep, in each relation\'s hue)</span></span>':"")+
  (runs.length>1?'<span class=lg><span class=mut>other runs, mean</span>'+runs.filter(function(r){return r!==run}).map(function(r,ri){return sw("var(--mut)",STY[1+ri%(STY.length-1)])+'<span>'+r.label+'</span>'}).join("")+'</span>':"")+
  (keys.length?key:"")+tabs+miss+'</div>';
 var tb=head.querySelector(".lg button")?head:null;if(tb)tb.onclick=function(e){var b=e.target.closest("button[data-b]");if(b){sel0=b.dataset.b;draw()}};
 // The panels are built once per layout and redrawn in place, so a card
 // open full screen stays open (expand.js).
 var sig=lay.panels.map(function(p){return p.join(":")}).join(",");
 grid.style.setProperty("--bcols",lay.cols);
 if(sig!==shape){shape=sig;[].forEach.call(grid.querySelectorAll(".bpanel"),function(c){c.remove()});
  var t=document.createElement("template");
  t.innerHTML=lay.panels.map(function(p){var k=D.pairs.find(function(x){return x.key===p[1]})||{};
   if(p[0]==="line")return card("bl-"+p[1],k.label,"each layer, mean over heads",p[1]===pairs[0].key?"biasline":"");
   if(p[0]==="heat")return card("bh-"+p[1],k.label,"latest, by layer and head",p[1]===pairs[0].key?"biasheat":"");
   if(p[0]==="layers")return card("bm","Mean by layer","latest, each relation","relations");
   return '<div id=lrslot></div>'}).join("");
  grid.insertBefore(t.content,grid.firstChild);
  var slot=$("lrslot");if(slot){if(lr)slot.replaceWith(lr);else slot.remove()}}
 pairs.forEach(function(k){var c=$("bl-"+k.key);if(c){var g=line(runs,run,k,lim),b=c.querySelector(".bsub");
   b.dataset.d=(runs.length>1?run.label+", ":"")+(((run.layers||{})[k.key]||[]).length?"each layer, mean over heads":"mean over layers and heads");
   b.textContent=b.dataset.d;c.querySelector(".bplot").innerHTML=g.svg;hover(c,g.hits)}
  var h=$("bh-"+k.key);if(h)h.querySelector(".bplot").innerHTML=lat[k.key]?heat(lat[k.key],k,amp):'<p class=mut>no matrix saved</p>';
  if(h)h.querySelector(".bsub").textContent=lat[k.key]?run.label+" at "+Math.round(lat.n)+" simulations":"no matrix saved"});
 var m=$("bm");if(m)m.querySelector(".bplot").innerHTML=keys.length?layers(lat,keys,{lo:Math.min(mlo-mp,0),hi:Math.max(mhi+mp,0)}):'<p class=mut>no matrix saved</p>'}
window.addEventListener("ftsel",draw);
// New data from the server (live.js): redraw with the same state.
window.addEventListener("ftdata",function(e){if(e.detail!=="bias")return;D=JSON.parse($("bdata").textContent);draw()});
draw();
})();
