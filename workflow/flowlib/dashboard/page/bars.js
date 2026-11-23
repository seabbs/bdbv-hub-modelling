// Grouped bars: one group per part (in `groups` order), one bar per model
// in each group, heights by value from a shared zero, in a W x H box with
// room below for the group labels. Returns the bars ({g, m, x, y, w, h,
// v}) and the scale's top value; a model with no value in a group gets no
// bar there.
function groupedBars(groups,models,W,H){
 var L=36,B=16,T=12,top=0,out=[];
 models.forEach(function(m){groups.forEach(function(g){var v=m.values[g];if(v!=null&&isFinite(v))top=Math.max(top,v)})});
 top=top>0?top*1.08:1;
 var gw=(W-L)/Math.max(1,groups.length),bw=Math.max(3,Math.min(18,(gw-6)/Math.max(1,models.length)));
 groups.forEach(function(g,gi){var x0=L+gi*gw+(gw-bw*models.length)/2;
  models.forEach(function(m,mi){var v=m.values[g];if(v==null||!isFinite(v))return;
   var h=Math.max(1,v/top*(H-B-T));out.push({g:g,m:m.id,x:x0+mi*bw,y:H-B-h,w:bw-1,h:h,v:v})})});
 return {bars:out,top:top,left:L,bottom:B,width:gw}}
if(typeof module!=="undefined")module.exports={groupedBars:groupedBars};
