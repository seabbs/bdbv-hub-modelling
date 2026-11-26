// The LoRA section: one even grid of equal panels (PW x PH, as the
// attention bias's), from flowlib.dashboard.level_loss_lora. The first
// row is the update's size against simulations trained, on one scale:
// each layer of time attention and of group attention (shallow light to
// deep dark in the kind's hue), each projection, and the output layer.
// The second is by layer: how far each layer moved since the previous
// checkpoint (time and group attention, one colour scale), then the last
// checkpoint's effective rank and the sizes of A and B. One run's detail
// at a time (the tabs above); every other selected run is one dashed line
// in the first row. One legend sits above the grid.
var LPW=320,LPH=180;
// loraLayout: the panels in grid order, "row:panel".
function loraLayout(){return ["row1:time","row1:group","row1:proj","row1:output",
 "row2:change-time","row2:change-group","row2:rank","row2:ab"]}
if(typeof module!=="undefined")module.exports={loraLayout:loraLayout};
else (function(){
var D=JSON.parse(document.getElementById("lodata").textContent),TIP=JSON.parse(document.getElementById("lotips").textContent);
var $=function(i){return document.getElementById(i)};
var KT="time attention",KG="group attention",PROJ=["q","k","v","o"],STY=["6 3","2 3","8 3 2 3"];
var sel0=null,built=false;
// The attention bias's layer ramp (bias.js), so depth reads alike.
var shade=window.layerShade||function(c){return c};
function fmt(v){return v===null||v===undefined?"n/a":String(+(+v).toPrecision(3))}
function nice(lo,hi,n){var st=(hi-lo)/(n||4)||1,m=Math.pow(10,Math.floor(Math.log10(st))),k=[1,2,2.5,5,10].find(function(x){return x*m>=st})*m,o=[];
 for(var v=Math.ceil(lo/k)*k;v<=hi+1e-9;v+=k)o.push(+v.toFixed(6));return o}
function sw(c,d){return '<svg width=22 height=8 viewBox="0 0 22 8"><line x1=0 x2=22 y1=4 y2=4 stroke="'+c+'" stroke-width=2'+(d?' stroke-dasharray="'+d+'"':"")+' /></svg>'}
function svgOpen(){return '<svg viewBox="0 0 '+LPW+' '+LPH+'" width="100%" role="img">'}
// frame: axes on [x0,x1] x [lo,hi].
function frame(lo,hi,x0,x1,xlab){var L=44,R=8,T=6,B=30,sy=function(y){return T+(1-(y-lo)/((hi-lo)||1))*(LPH-T-B)},sx=function(x){return L+(x-x0)/((x1-x0)||1)*(LPW-L-R)},s="";
 nice(lo,hi,3).forEach(function(v){if(v>=lo&&v<=hi)s+='<line x1="'+L+'" x2="'+(LPW-R)+'" y1="'+sy(v).toFixed(1)+'" y2="'+sy(v).toFixed(1)+'" class=grid /><text x="'+(L-4)+'" y="'+(sy(v)+3).toFixed(1)+'" text-anchor=end class=ax>'+fmt(v)+'</text>'});
 nice(x0,x1,3).forEach(function(v){if(v>=x0&&v<=x1)s+='<text x="'+sx(v).toFixed(1)+'" y="'+(LPH-B+13)+'" text-anchor=middle class=ax>'+v+'</text>'});
 s+='<text x="'+((L+LPW-R)/2)+'" y="'+(LPH-3)+'" text-anchor=middle class=ax>'+xlab+'</text>';
 return {sx:sx,sy:sy,s:s}}
// lines: each [points, colour, dash, label] on one frame; returns the svg
// and its hover points.
function lines(f,ls){var s=svgOpen()+f.s,hits=[];
 ls.forEach(function(l){var pts=l[0].filter(function(p){return p[1]!==null&&p[1]!==undefined});if(!pts.length)return;
  s+='<polyline points="'+pts.map(function(p){return f.sx(p[0]).toFixed(1)+","+f.sy(p[1]).toFixed(1)}).join(" ")+'" fill=none style="stroke:'+l[1]+'" stroke-width=1.5 stroke-linejoin=round'+(l[2]?' stroke-dasharray="'+l[2]+'"':"")+'><title>'+l[3]+'</title></polyline>';
  pts.forEach(function(p){s+='<circle cx="'+f.sx(p[0]).toFixed(1)+'" cy="'+f.sy(p[1]).toFixed(1)+'" r=1.6 style="fill:'+l[1]+'" />';
   hits.push({x:f.sx(p[0]),y:f.sy(p[1]),c:l[1],t:l[3]+(l[4]?l[4](p[0]):"")+": "+fmt(p[1])})})});
 return {svg:s+'<circle class=bhit r=4 cx=-9 cy=-9 stroke="var(--card)" stroke-width=2 style="display:none" /></svg>',hits:hits}}
// hover: the point nearest the pointer is ringed and named in the
// panel's subtitle.
function hover(card,hits){var svg=card.querySelector("svg"),sub=card.querySelector(".bsub"),dot=svg.querySelector(".bhit");
 svg.onmousemove=function(e){var r=svg.getBoundingClientRect(),mx=(e.clientX-r.left)*LPW/r.width,my=(e.clientY-r.top)*LPH/r.height,best=null,bd=Infinity;
  hits.forEach(function(h){var d=Math.pow(h.x-mx,2)+Math.pow(h.y-my,2);if(d<bd){bd=d;best=h}});
  if(!best||bd>900){svg.onmouseleave();return}
  dot.setAttribute("cx",best.x.toFixed(1));dot.setAttribute("cy",best.y.toFixed(1));dot.style.fill=best.c;dot.style.display="";sub.textContent=best.t};
 svg.onmouseleave=function(){dot.style.display="none";sub.textContent=sub.dataset.d}}
function at(n){return " at "+Math.round(n)+" simulations"}
function rms(xs){var v=xs.filter(function(x){return x!==null});return v.length?Math.sqrt(v.reduce(function(a,b){return a+b*b},0)/v.length):null}
// heat: one kind's change since the previous checkpoint, layer (row) x
// checkpoint (column), in the kind's hue, on the shared scale `amp`.
function heat(run,kind,amp){var ch=run.change[kind],cols=[];ch.forEach(function(r,i){if(r)cols.push(i)});
 if(!cols.length)return '<p class=mut>one checkpoint so far: nothing to compare</p>';
 var nr=run.layers,nc=cols.length,L=44,T=6,B=30,w=(LPW-L-8)/nc,h=(LPH-T-B)/nr,c=D.col[kind],s=svgOpen();
 cols.forEach(function(ci,j){ch[ci].forEach(function(v,i){if(v===null)return;var a=Math.min(1,v/amp);
  s+='<rect x="'+(L+j*w).toFixed(1)+'" y="'+(T+i*h).toFixed(1)+'" width="'+(w-1).toFixed(1)+'" height="'+(h-1).toFixed(1)+'" style="fill:'+c+'" fill-opacity="'+(0.06+0.94*a).toFixed(2)+'"><title>'+kind+', layer '+i+', '+run.n[ci-1]+' to '+run.n[ci]+' simulations: '+fmt(v)+'</title></rect>'})});
 [0,nr-1].forEach(function(i){s+='<text x="'+(L-4)+'" y="'+(T+i*h+h*0.75).toFixed(1)+'" text-anchor=end class=ax>'+i+'</text>'});
 s+='<text x="'+(L-4)+'" y="'+(T+nr*h/2+3).toFixed(1)+'" text-anchor=end class=ax>layer</text>';
 var step=Math.ceil(nc/5);cols.forEach(function(ci,j){if(j%step==0||j==nc-1)s+='<text x="'+(L+j*w+w/2).toFixed(1)+'" y="'+(LPH-B+13)+'" text-anchor=middle class=ax>'+run.n[ci]+'</text>'});
 return s+'<text x="'+((L+LPW-8)/2)+'" y="'+(LPH-3)+'" text-anchor=middle class=ax>simulations trained (to)</text></svg>'}
function card(id,title,info){return '<div class="card bpanel" id="'+id+'"><h3>'+title+(TIP[info]||"")+'</h3><div class="leg bsub"></div><div class=bplot></div></div>'}
var TITLES={"row1:time":["Time attention","lorasize"],"row1:group":["Group attention",""],"row1:proj":["By projection","loraproj"],"row1:output":["Output layer",""],
 "row2:change-time":["Time attention: change","lorachange"],"row2:change-group":["Group attention: change",""],"row2:rank":["Effective rank","lorarank"],"row2:ab":["Size of A and B","loraab"]};
function put(id,sub,body,hits){var c=$("lo-"+id.replace(":","-"));if(!c)return;var b=c.querySelector(".bsub");b.dataset.d=sub;b.textContent=sub;
 c.querySelector(".bplot").innerHTML=body;if(hits)hover(c,hits)}
function draw(){
 var on=ftSel.get(),runs=D.runs.filter(function(r){return on.indexOf(r.key)>=0}),grid=$("loragrid"),head=$("lorahead");
 var none=(D.missing||[]).filter(function(r){return on.indexOf(r.key)>=0}).map(function(r){return r.label});
 var miss=none.length?'<span class=mut>no adapter saved yet: '+none.join(", ")+'</span>':"";
 if(!runs.length){head.innerHTML='<p class=mut>none of the selected runs has a saved adapter</p>'+miss;grid.innerHTML="";built=false;return}
 if(!sel0||!runs.some(function(r){return r.key==sel0}))sel0=runs[runs.length-1].key;
 var run=runs.find(function(r){return r.key==sel0}),others=runs.filter(function(r){return r!==run}),nl=run.layers;
 // Shared scales: the first row's update size, the change heat maps.
 var ys=[],x1=1,amp=0;
 runs.forEach(function(r){x1=Math.max(x1,r.n[r.n.length-1]);
  [KT,KG].forEach(function(k){r.size[k].forEach(function(row){row.forEach(function(v){ys.push(v)})})});
  PROJ.forEach(function(p){(r.proj[p]||[]).forEach(function(v){if(v!==null)ys.push(v)})});
  r.output.forEach(function(v){if(v!==null)ys.push(v)})});
 [KT,KG].forEach(function(k){run.change[k].forEach(function(row){(row||[]).forEach(function(v){if(v!==null)amp=Math.max(amp,v)})})});
 var hi=(ys.length?Math.max.apply(null,ys):1)*1.08||1,lead=runs.length>1?run.label+", ":"";
 var tabs=runs.length>1?'<span class=lg><span class=mut>layers of</span>'+runs.map(function(r){return '<button class="tog'+(r.key==sel0?" on":"")+'" data-l="'+r.key+'" style="border-left:4px solid '+r.color+'">'+r.label+'</button>'}).join("")+'</span>':"";
 head.innerHTML='<div class=leg2>'+[KT,KG,"output layer"].map(function(k){return '<span><i class=sw2 style="background:'+D.col[k]+'"></i>'+k+'</span>'}).join("")+
  PROJ.map(function(p){return '<span><i class=sw2 style="background:'+D.col[p]+'"></i>'+p+'</span>'}).join("")+
  (nl>1?'<span class=lg><span class=mut>layer 0</span><svg width=72 height=8 viewBox="0 0 72 8">'+[0,1,2,3,4,5].map(function(i){return '<rect x="'+(i*12)+'" y=0 width=11 height=8 style="fill:'+shade("var(--fg)",i,6)+'" />'}).join("")+'</svg><span class=mut>'+(nl-1)+' (shallow to deep)</span></span>':"")+
  (others.length?'<span class=lg><span class=mut>other runs</span>'+others.map(function(r,i){return sw("var(--mut)",STY[i%STY.length])+'<span>'+r.label+'</span>'}).join("")+'</span>':"")+
  '<span class=lg><span class=mut>change 0</span><svg width=60 height=8 viewBox="0 0 60 8">'+[0,1,2,3,4].map(function(i){return '<rect x="'+(i*12)+'" y=0 width=11 height=8 style="fill:var(--fg)" fill-opacity="'+(0.06+0.94*i/4).toFixed(2)+'" />'}).join("")+'</svg><span class=mut>'+fmt(amp)+'</span></span>'+tabs+miss+'</div>';
 var tb=head.querySelector("button[data-l]");if(tb)head.onclick=function(e){var b=e.target.closest("button[data-l]");if(b){sel0=b.dataset.l;draw()}};
 // The panels are built once and redrawn in place, so a card open full
 // screen stays open (expand.js).
 grid.style.setProperty("--bcols",4);
 if(!built){built=true;grid.innerHTML=loraLayout().map(function(p){return card("lo-"+p.replace(":","-"),TITLES[p][0],TITLES[p][1])}).join("")}
 function f1(){return frame(0,hi,0,x1,"simulations trained")}
 [KT,KG].forEach(function(k,ki){var c=D.col[k],ls=[];
  for(var i=0;i<nl;i++)ls.push([run.n.map(function(n,j){return [n,run.size[k][j][i]]}),shade(c,i,nl),"",lead+"layer "+i,at]);
  others.forEach(function(r,oi){ls.push([r.n.map(function(n,j){return [n,rms(r.size[k][j])]}),"var(--mut)",STY[oi%STY.length],r.label+", root mean square over layers",at])});
  var g=lines(f1(),ls);put(ki?"row1:group":"row1:time",lead+"each layer's update size",g.svg,g.hits)});
 var g=lines(f1(),PROJ.map(function(p){return [run.n.map(function(n,j){return [n,run.proj[p][j]]}),D.col[p],"",lead+p,at]}));
 put("row1:proj",lead+"root mean square over layers",g.svg,g.hits);
 var hasOut=runs.some(function(r){return r.output.some(function(v){return v!==null})});
 g=lines(f1(),[[run.n.map(function(n,j){return [n,run.output[j]]}),D.col["output layer"],"",lead+"output layer",at]].concat(others.map(function(r,oi){return [r.n.map(function(n,j){return [n,r.output[j]]}),"var(--mut)",STY[oi%STY.length],r.label,at]})));
 put("row1:output",hasOut?"the output patch embedding's update size":"no output layer adapter",hasOut?g.svg:"",hasOut?g.hits:null);
 var last=run.n[run.n.length-1];
 [KT,KG].forEach(function(k,ki){put(ki?"row2:change-group":"row2:change-time",lead+"over the update's size, by layer",heat(run,k,amp||1),null)});
 var lx=[0,Math.max(nl-1,1)],byl=function(k,key,dash,name){return [run[key][k].map(function(v,i){return [i,v]}),D.col[k],dash,lead+name+k+", layer",function(i){return " "+i}]};
 var rk=[];[KT,KG].forEach(function(k){run.rank[k].forEach(function(v){if(v!==null)rk.push(v)})});
 g=lines(frame(0,Math.max(1,Math.ceil(rk.length?Math.max.apply(null,rk):1)),lx[0],lx[1],"layer"),[byl(KT,"rank","",""),byl(KG,"rank","","")]);
 put("row2:rank",lead+"mean over q, k, v, o at "+last,g.svg,g.hits);
 var ab=[];[KT,KG].forEach(function(k){["a","b"].forEach(function(s){run[s][k].forEach(function(v){if(v>0)ab.push(v)})})});
 var ahi=(ab.length?Math.max.apply(null,ab):1)*1.08;
 g=lines(frame(0,ahi,lx[0],lx[1],"layer"),[byl(KT,"a","","A, "),byl(KG,"a","","A, "),byl(KT,"b","6 3","B, "),byl(KG,"b","6 3","B, ")]);
 put("row2:ab",lead+"A solid, B dashed, at "+last,g.svg,g.hits)}
window.addEventListener("ftsel",draw);
// New data from the server (live.js): redraw with the same state.
window.addEventListener("ftdata",function(e){if(e.detail!=="lora")return;D=JSON.parse($("lodata").textContent);draw()});
draw();
})();
