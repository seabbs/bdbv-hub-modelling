// Axis helpers shared by the loss charts.
// thinLabels: the indices of labels at pixel positions `xs` (ascending)
// to draw, left to right, each at least `gap` from the last one drawn.
function thinLabels(xs,gap){var out=[],last=-Infinity;
 xs.forEach(function(x,i){if(x-last>=gap){out.push(i);last=x}});return out}
// splitGaps: a line's [x, y] points cut into runs where the step to the
// next point is over four times the median step, so a line never bridges
// a stretch with no values.
function splitGaps(p){if(p.length<5)return [p];
 var d=[];for(var i=1;i<p.length;i++)d.push(p[i][0]-p[i-1][0]);
 var s=d.slice().sort(function(a,b){return a-b}),m=s[Math.floor(s.length/2)],out=[[p[0]]];
 for(var j=1;j<p.length;j++){if(m>0&&d[j-1]>4*m)out.push([]);out[out.length-1].push(p[j])}
 return out}
if(typeof module!=="undefined")module.exports={thinLabels:thinLabels,splitGaps:splitGaps};
