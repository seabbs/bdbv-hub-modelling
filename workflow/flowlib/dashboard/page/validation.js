// Validation forecasts: a run's checkpoints' (and base Chronos-2's)
// forecasts of a sample of its held-out blocks against the truth, one
// block and row at a time, as the German nowcasts are shown. The index
// and each block come from the server (validation/<run>/index.json and
// validation/<run>/<block>.json), relative to the page.
(function(){
var D=JSON.parse(document.getElementById("vdata").textContent),$=function(i){return document.getElementById(i)};
var S={run:null,idx:null,block:null,row:0,on:{},data:null};
function esc(t){return String(t).replace(/[&<>"]/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]})}
function fmt(v){return v==null?"–":String(+(+v).toPrecision(3))}
function colour(key){if(key==="base")return D.base;var ms=S.idx.models.filter(function(m){return m.key!=="base"});
 var i=ms.findIndex(function(m){return m.key===key});return D.palette[i%D.palette.length]}
function label(m){return m.key==="base"?m.label:m.label+" ("+m.n+" simulations)"}
function controls(){var runs=D.runs,b=S.idx.blocks;
 $("vrun").innerHTML=runs.map(function(r){return '<option value="'+esc(r.key)+'"'+(r.key===S.run?" selected":"")+'>'+esc(r.label)+'</option>'}).join("");
 $("vblock").innerHTML=b.map(function(x){return '<option value="'+x.block+'"'+(x.block===S.block?" selected":"")+'>block '+x.block+' · '+esc(x.source)+' · '+esc(x.mix)+(x.cut!=null?' · cut '+esc(x.cut):'')+'</option>'}).join("");
 var blk=b.find(function(x){return x.block===S.block})||{rows:[]};
 $("vrow").innerHTML=blk.rows.map(function(n,i){return '<option value="'+i+'"'+(i===S.row?" selected":"")+'>'+esc(n)+'</option>'}).join("");
 $("vmodels").innerHTML=S.idx.models.map(function(m){var on=S.on[m.key];return '<button class="tog'+(on?" on":"")+'" data-m="'+esc(m.key)+'" style="border-left:4px solid '+colour(m.key)+(on?';background:'+colour(m.key)+';border-color:'+colour(m.key):'')+'">'+esc(label(m))+'</button>'}).join("")}
function draw(){var el=$("valplot");if(!S.data){el.innerHTML='<p class=mut>'+(S.idx&&!S.idx.blocks.length?"No validation forecasts for this run yet: they are written after each checkpoint.":"Loading…")+'</p>';return}
 var r=S.data.rows[S.row]||S.data.rows[0],hist=r.history,truth=r.truth,nh=hist.length,h=truth.length;
 var W=Math.max(320,el.clientWidth-8),H=300,L=48,R=10,T=8,B=30,keys=S.idx.models.filter(function(m){return S.on[m.key]&&S.data.q[m.key]}).map(function(m){return m.key});
 var ys=[];hist.concat(truth).forEach(function(v){if(v!=null)ys.push(v)});
 keys.forEach(function(k){var q=S.data.q[k][S.row];[0,3,6].forEach(function(j){q[j].forEach(function(v){if(v!=null)ys.push(v)})})});
 if(!ys.length){el.innerHTML='<p class=mut>No values.</p>';return}
 var y0=Math.min(0,Math.min.apply(null,ys)),y1=Math.max.apply(null,ys)||1,x0=-nh,x1=h-1;
 var sx=function(x){return L+(x-x0)/(x1-x0)*(W-L-R)},sy=function(y){return T+(1-(y-y0)/(y1-y0))*(H-T-B)};
 var s='<svg viewBox="0 0 '+W+' '+H+'" width="100%" role="img">';
 for(var k=0;k<=4;k++){var v=y0+(y1-y0)*k/4;s+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(v)+'" y2="'+sy(v)+'" class=grid /><text x="'+(L-4)+'" y="'+(sy(v)+3)+'" text-anchor=end class=ax>'+fmt(v)+'</text>'}
 [x0,Math.round(x0/2),0,S.data.nowcast,x1].forEach(function(v){s+='<text x="'+sx(v)+'" y="'+(H-B+13)+'" text-anchor=middle class=ax>'+v+'</text>'});
 s+='<text x="'+((L+W-R)/2)+'" y="'+(H-2)+'" text-anchor=middle class=ax>steps from the block\'s first forecast step</text>';
 var cut=sx(S.data.nowcast-0.5);s+='<line x1="'+cut+'" x2="'+cut+'" y1="'+T+'" y2="'+(H-B)+'" class=mark><title>the cut-off: nowcast steps before it, forecast steps after</title></line>';
 function path(xs,vs){var d="",pen=false;vs.forEach(function(v,i){if(v==null){pen=false;return}d+=(pen?"L":"M")+sx(xs[i]).toFixed(1)+" "+sy(v).toFixed(1);pen=true});return d}
 function band(lo,hi){var pts=[],i;for(i=0;i<h;i++)if(lo[i]!=null&&hi[i]!=null)pts.push(sx(i).toFixed(1)+","+sy(hi[i]).toFixed(1));for(i=h-1;i>=0;i--)if(lo[i]!=null&&hi[i]!=null)pts.push(sx(i).toFixed(1)+","+sy(lo[i]).toFixed(1));return pts.join(" ")}
 var steps=truth.map(function(_,i){return i});
 keys.forEach(function(k){var q=S.data.q[k][S.row],c=colour(k);
  s+='<polygon points="'+band(q[0],q[6])+'" fill="'+c+'" opacity=".12" /><polygon points="'+band(q[2],q[4])+'" fill="'+c+'" opacity=".22" />';
  s+='<path d="'+path(steps,q[3])+'" fill=none stroke="'+c+'" stroke-width=2 />'});
 s+='<path d="'+path(hist.map(function(_,i){return i-nh}),hist)+'" fill=none stroke="var(--fg)" stroke-width=1.5 />';
 truth.forEach(function(v,i){if(v!=null)s+='<circle cx="'+sx(i)+'" cy="'+sy(v)+'" r=2.2 fill="var(--fg)"><title>step '+i+': truth '+fmt(v)+keys.map(function(k){var q=S.data.q[k][S.row];return "; "+k+" "+fmt(q[3][i])+" (50%: "+fmt(q[2][i])+" to "+fmt(q[4][i])+")"}).join("")+'</title></circle>'});
 el.innerHTML=s+'</svg>';
 $("vleg").innerHTML='<span><i style="background:var(--fg)"></i>history and truth</span>'+keys.map(function(k){var m=S.idx.models.find(function(x){return x.key===k});return '<span><i style="background:'+colour(k)+'"></i>'+esc(label(m))+'</span>'}).join("")+'<span class=mut>'+esc(r.name)+(r.level?" ("+esc(r.level)+")":"")+'; bands 50% and 95%</span>'}
function loadBlock(){S.data=null;draw();if(S.block==null)return;
 fetch("validation/"+encodeURIComponent(S.run)+"/"+S.block+".json").then(function(r){return r.ok?r.json():null}).then(function(d){S.data=d;if(d&&S.row>=d.rows.length)S.row=0;controls();draw()}).catch(function(){})}
function loadRun(key){S.run=key;S.idx={models:[],blocks:[]};S.data=null;draw();
 fetch("validation/"+encodeURIComponent(key)+"/index.json").then(function(r){return r.ok?r.json():{models:[],blocks:[]}}).then(function(ix){
  S.idx=ix;S.on={};if(ix.models.length){S.on[ix.models[0].key]=true;S.on[ix.models[ix.models.length-1].key]=true}
  if(!ix.blocks.some(function(b){return b.block===S.block}))S.block=ix.blocks.length?ix.blocks[0].block:null;
  S.row=0;controls();loadBlock()}).catch(function(){draw()})}
$("vrun").onchange=function(){loadRun(this.value)};
$("vblock").onchange=function(){S.block=+this.value;S.row=0;loadBlock()};
$("vrow").onchange=function(){S.row=+this.value;draw()};
$("vmodels").onclick=function(e){var b=e.target.closest("button[data-m]");if(!b)return;S.on[b.dataset.m]=!S.on[b.dataset.m];controls();draw()};
var rsz=null;window.addEventListener("resize",function(){clearTimeout(rsz);rsz=setTimeout(draw,120)});
if(D.runs.length)loadRun(D.runs[0].key);
})();
