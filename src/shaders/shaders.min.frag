// src\shaders\preprocessed.scene.frag#version 150

#define ENABLE_STOCHASTIC_MOTION_BLUR

const vec2 iResolution=vec2(1920,1080);
uniform float iTime;
in vec3 camPos,camTa;
in float camMotoSpace,camFoV,camProjectionRatio,camFishEye,camShowDriver;
in vec2 spline[13];
out vec4 fragColor;
float PIXEL_ANGLE=camFoV/iResolution.x;

#define ZERO(iTime)min(0,int(iTime))

const float PI=acos(-1.);
struct light{vec3 p0;vec3 p1;vec3 color;float cosAngle;float collimation;float luminance;};
struct material{int type;vec3 color;float roughness;};
vec3 cookTorrance(float roughness,vec3 NcrossH,float VdotH,float NdotL,float NdotV)
{
  roughness*=roughness;
  roughness*=roughness;
  float distribution=dot(NcrossH,NcrossH)*(1.-roughness)+roughness;
  VdotH=1.-VdotH;
  VdotH*=VdotH*VdotH*VdotH*VdotH;
  NcrossH=VdotH+vec3(.04)*(1.-VdotH);
  return NcrossH*(roughness/(PI*distribution*distribution)*(2.*NdotL/max(1e-8,NdotL+sqrt(NdotL*NdotL*(1.-roughness)+roughness))*(2.*NdotV/max(1e-8,NdotV+sqrt(NdotV*NdotV*(1.-roughness)+roughness))))*.25/max(1e-8,NdotV*NdotL));
}
vec3 coneLightContribution(material m,light l,vec3 p,vec3 N,vec3 V)
{
  p=l.p0-p;
  float d0=length(p);
  p/=d0;
  float NdotL=dot(N,p);
  if(NdotL<=0.)
    return vec3(0);
  vec3 radiance=l.color*l.luminance*smoothstep(l.cosAngle,mix(l.cosAngle,1.,.5),dot(p,-l.p1))*((1.+l.collimation)/(l.collimation+d0*d0))*NdotL;
  if(m.type==3)
    return.1*radiance*m.color*pow(clamp(dot(V,p),0.,1.),1e3)*501.;
  p=normalize(p+V);
  return radiance*(m.color+cookTorrance(m.roughness,cross(N,p),clamp(dot(V,p),0.,1.),NdotL,clamp(dot(N,V),0.,1.)));
}
vec3 rodLightContribution(material m,light l,vec3 p,vec3 N,vec3 V)
{
  vec3 L0=l.p0-p;
  p=l.p1-p;
  float d0=length(L0),d1=length(p);
  d0=2.*clamp((dot(N,L0)/d0+dot(N,p)/d1)/2.,0.,1.)/(d0*d1+dot(L0,p));
  if(d0<=0.)
    return vec3(0);
  vec3 irradiance=l.color*l.luminance*d0,Ld=l.p1-l.p0,R=reflect(-V,N);
  d0=dot(R,Ld);
  Ld=normalize(mix(L0,p,clamp((dot(R,L0)*d0-dot(L0,Ld))/(dot(Ld,Ld)-d0*d0),0.,1.)));
  d0=clamp(dot(N,Ld),0.,1.);
  if(m.type==3)
    return irradiance*d0*m.color*pow(clamp(dot(V,Ld),0.,1.),1e2)*51.;
  Ld=normalize(Ld+V);
  return irradiance*(m.color+cookTorrance(m.roughness,cross(N,Ld),clamp(dot(V,Ld),0.,1.),d0,clamp(dot(N,V),0.,1.)));
}
float hash21(vec2 xy)
{
  return fract(sin(dot(xy,vec2(12.9898,78.233)))*43758.5453);
}
float hash31(vec3 xyz)
{
  return hash21(vec2(hash21(xyz.xy),xyz.z));
}
float valueNoise(vec2 p)
{
  vec2 p00=floor(p);
  p-=p00;
  p=p*p*(3.-2.*p);
  return mix(mix(hash21(p00),hash21(p00+vec2(1,0)),p.x),mix(hash21(p00+vec2(0,1)),hash21(p00+vec2(1)),p.x),p.y);
}
float fBm(vec2 p,int iterations,float weight_param,float frequency_param)
{
  float v=0.,weight=1.,frequency=1.,offset=0.;
  for(int i=ZERO(iTime);i<iterations;++i)
    {
      float noise=valueNoise(p*frequency+offset)*2.-1.;
      v+=weight*noise;
      weight*=clamp(weight_param,0.,1.);
      frequency*=1.+2.*clamp(frequency_param,0.,1.);
      offset+=1.;
    }
  return v;
}
float smin(float a,float b,float k)
{
  k*=1./(1.-sqrt(.5));
  return max(k,min(a,b))-length(max(k-vec2(a,b),0.));
}
float Box2(vec2 p,vec2 size,float corner)
{
  p=abs(p)-size+corner;
  return length(max(p,0.))+min(max(p.x,p.y),0.)-corner;
}
float Box3(vec3 p,vec3 size,float corner)
{
  p=abs(p)-size+corner;
  return length(max(p,0.))+min(max(max(p.x,p.y),p.z),0.)-corner;
}
float Ellipsoid(vec3 p,vec3 r)
{
  float k0=length(p/r);
  return k0*(k0-1.)/length(p/(r*r));
}
float Segment3(vec3 p,vec3 a,vec3 b,out float h)
{
  p-=a;
  a=b-a;
  h=clamp(dot(p,a)/dot(a,a),0.,1.);
  return length(p-a*h);
}
float Capsule(vec3 p,float h,float r)
{
  p.y+=clamp(-p.y,0.,h);
  return length(p)-r;
}
float Torus(vec3 p,vec2 t)
{
  return length(vec2(length(p.xz)-t.x,p.y))-t.y;
}
mat2 Rotation(float angle)
{
  float c=cos(angle);
  angle=sin(angle);
  return mat2(c,angle,-angle,c);
}
float DistanceFromAABB(vec2 p,vec4 aabb)
{
  return Box2(p-(aabb.xy+aabb.zw)/2.,(aabb.zw-aabb.xy)/2.,0.);
}
vec2 Bezier(vec2 A,vec2 B,vec2 C,float t)
{
  return mix(mix(A,B,t),mix(B,C,t),t);
}
vec4 FindCubicRoots(float a,float b,float c)
{
  float p=b-a*a/3.,p3=p*p*p;
  b=a*(2.*a*a-9.*b)/27.+c;
  c=b*b+4.*p3/27.;
  a=-a/3.;
  if(c>=0.)
    {
      float z=sqrt(c);
      vec2 x=(vec2(z,-z)-b)/2.;
      x=sign(x)*pow(abs(x),vec2(1)/3.);
      return vec4(a+x.x+x.y,0,0,1);
    }
  p3=acos(-sqrt(-27./p3)*b/2.)/3.;
  b=cos(p3);
  p3=sin(p3)*sqrt(3.);
  return vec4(vec3(b+b,-p3-b,p3-b)*sqrt(-p/3.)+a,3);
}
float GetWinding(vec2 a,vec2 b)
{
  return 2.*step(a.x*b.y,a.y*b.x)-1.;
}
vec2 BezierSDF(vec2 A,vec2 B,vec2 C,vec2 p)
{
  vec2 a=B-A;
  C+=A-B*2.;
  B=a*2.;
  A-=p;
  vec3 k=vec3(3.*dot(a,C),2.*dot(a,a)+dot(A,C),dot(A,a))/dot(C,C);
  vec4 t=FindCubicRoots(k.x,k.y,k.z);
  a=clamp(t.xy,0.,1.);
  p=A+(B+C*a.x)*a.x;
  float d1=dot(p,p);
  a=A+(B+C*a.y)*a.y;
  float d2=dot(a,a);
  t=d1<d2?
    vec4(d1,t.x,p):
    vec4(d2,t.y,a);
  return vec2(GetWinding(t.zw,2.*C*t.y+B)*sqrt(t.x),t.y);
}
float BezierCurveLengthAt(vec2 A,vec2 B,vec2 C,float t)
{
  C=2.*(A-2.*B+C);
  A=2.*(B-A);
  float k1=dot(C,C),k4=2.*dot(C,A)*.5/k1,k5=dot(A,A)/k1-k4*k4;
  A=vec2(0,t);
  B=sqrt((A+k4)*(A+k4)+k5);
  A=sqrt(k1)*.5*((k4+A)*B+k5*log(abs(k4+A+B)));
  return A.y-A.x;
}
vec4 BezierAABB(vec2 A,vec2 B,vec2 C)
{
  vec2 mi=min(A,C),ma=max(A,C);
  if(B.x<mi.x||B.x>ma.x||B.y<mi.y||B.y>ma.y)
    {
      vec2 t=clamp((A-B)/(A-2.*B+C),0.,1.),s=1.-t;
      t=s*s*A+2.*s*t*B+t*t*C;
      mi=min(mi,t);
      ma=max(ma,t);
    }
  return vec4(mi,ma);
}
vec2 MinDist(vec2 d1,vec2 d2)
{
  return d1.x<d2.x?
    d1:
    d2;
}
void setupCamera(vec2 uv,vec3 cameraPosition,vec3 cameraTarget,float projectionRatio,float camFishEye,out vec3 ro,out vec3 rd)
{
  vec3 cameraUp=vec3(0,1,0);
  cameraTarget=normalize(cameraTarget-cameraPosition);
  if(abs(dot(cameraTarget,cameraUp))>.99)
    cameraUp=vec3(1,0,0);
  vec3 cameraRight=normalize(cross(cameraTarget,cameraUp));
  cameraUp=normalize(cross(cameraRight,cameraTarget));
  uv*=mix(1.,length(uv),camFishEye);
  ro=cameraPosition;
  rd=normalize(cameraTarget*projectionRatio+uv.x*cameraRight+uv.y*cameraUp);
}
bool IsMoto(float mid)
{
  return mid>=1.&&mid<=8.;
}
vec3 nightHorizonLight=.01*vec3(.07,.1,1),moonLightColor=vec3(.2,.8,1),moonDirection=normalize(vec3(-1,.3,.4));
vec3 sky(vec3 V)
{
  vec3 clearest=vec3(34,728,1910);
  float direction=clamp(dot(V,normalize(vec3(0,1,.25))),0.,1.);
  vec3 color=mix(vec3(126,387,728),mix(vec3(990,1527,1297),mix(vec3(332,1190,1777),mix(clearest,vec3(1,4.6,97),pow(direction,.5)),smoothstep(0.,.3,mix(V.y,direction,.5))),smoothstep(0.,.15,mix(V.y,direction,.2))),smoothstep(.05,.15,V.y));
  direction=clamp(dot(V,moonDirection),0.,1.);
  float dmoon=2.*fwidth(direction);
  direction=smoothstep(-dmoon,dmoon,direction-.9999);
  if(direction>0.)
    {
      float pattern=smoothstep(-.5,.5,fBm(V.xy*1e2,4,.65,.7)+.13);
      color=mix(color,clearest*5.*mix(1.,2.,pattern),.9*direction);
    }
  dmoon=fBm(.015*iTime+V.xz/(.01+V.y)*.5,5,.55,.7);
  dmoon=smoothstep(0.,1.,dmoon+1.);
  color=color*mix(.1,1.,pow(dmoon,2.))*smoothstep(-.1,0.,V.y);
  return color/1.97e5;
}
vec3 cityLights(vec2 p)
{
  vec3 ctex=vec3(0);
  for(int i=0;i<3;i++)
    {
      float fi=float(i);
      vec2 xp=p*Rotation(max(fi-3.,0.)*.5)*(1.+fi*.3),mp=mod(xp,10.)-5.;
      fi=smoothstep(.6+fi*.1,0.,min(abs(mp.x),abs(mp.y)))*max(smoothstep(.7+fi*.1,.5,length(mod(p,2.)-1.))*smoothstep(.5,.7,valueNoise(p)-.15),pow(valueNoise(xp*.5),10.));
      ctex+=valueNoise(xp*.5)*mix(mix(vec3(.56,.32,.18)*min(fi,.5)*2.,vec3(.88,.81,.54),max(fi-.5,0.)*2.),mix(vec3(.45,.44,.6)*min(fi,.5)*2.,vec3(.8,.89,.93),max(fi-.5,0.)*2.),step(.5,valueNoise(p*2.)));
    }
  return ctex*5.;
}
vec4 splineAABB,splineSegmentAABBs[6];
float splineSegmentDistances[6];
void ComputeBezierSegmentsLengthAndAABB()
{
  float splineLength=0.;
  splineAABB=vec4(1e6,1e6,-1e6,-1e6);
  for(int i=ZERO(iTime);i<12;i+=2)
    {
      vec2 A=spline[i],B=spline[i+1],C=spline[i+2];
      splineSegmentDistances[i/2]=splineLength;
      splineLength+=BezierCurveLengthAt(A,B,C,1.);
      vec4 AABB=BezierAABB(A,B,C);
      splineSegmentAABBs[i/2]=AABB;
      splineAABB.xy=min(splineAABB.xy,AABB.xy);
      splineAABB.zw=max(splineAABB.zw,AABB.zw);
    }
}
vec4 ToSplineLocalSpace(vec2 p,float splineWidth)
{
  vec4 splineUV=vec4(1e6,0,0,0);
  if(DistanceFromAABB(p,splineAABB)>splineWidth)
    return splineUV;
  for(int i=ZERO(iTime);i<12;i+=2)
    {
      vec2 A=spline[i],B=spline[i+1],C=spline[i+2];
      if(DistanceFromAABB(p,BezierAABB(A,B,C))>splineWidth)
        continue;
      B=mix(B+vec2(1e-4),B,abs(sign(B*2.-A-C)));
      vec2 bezierSDF=BezierSDF(A,B,C,p);
      if(abs(bezierSDF.x)<abs(splineUV.x))
        {
          float lengthInSegment=BezierCurveLengthAt(A,B,C,clamp(bezierSDF.y,0.,1.))+splineSegmentDistances[i/2];
          splineUV=vec4(bezierSDF.x,clamp(bezierSDF.y,0.,1.),lengthInSegment,float(i));
        }
    }
  return splineUV;
}
vec2 GetPositionOnCurve(float t)
{
  t*=splineSegmentDistances[5];
  int segmentIndex=0;
  for(int i=ZERO(iTime);i<5;++i)
    if(splineSegmentDistances[i]<=t&&splineSegmentDistances[i+1]>t)
      {
        segmentIndex=i;
        break;
      }
  float segmentStartLength=splineSegmentDistances[segmentIndex],segmentEndLength=splineSegmentDistances[segmentIndex+1];
  vec2 A=spline[segmentIndex*2],B=spline[segmentIndex*2+1],C=spline[segmentIndex*2+2];
  return Bezier(A,B,C,(t-segmentStartLength)/(segmentEndLength-segmentStartLength));
}
vec2 GetPositionOnSpline(vec2 spline_t_and_index)
{
  int i=int(spline_t_and_index.y);
  return Bezier(spline[i],spline[i+1],spline[i+2],spline_t_and_index.x);
}
vec3 roadWidthInMeters=vec3(4,8,8);
vec3 roadPattern(vec2 uv)
{
  vec2 params=vec2(.7,0),t1b=vec2(6.5,1.5),t3=vec2(26./6.,3),t3b=vec2(26,20),continuous=vec2(100);
  t3b=vec2(13,3);
  if(params.x>.25)
    t3b=t1b;
  if(params.x>.5)
    t3b=t3;
  if(params.x>.75)
    t3b=continuous;
  continuous=vec2(6.5,3);
  params=vec2(fract(uv.x/t3b.x)*t3b.x,uv.y-floor(clamp(uv.y,0.,3.5)/3.5)*3.5);
  t1b=vec2(fract((uv.x+.4)/continuous.x)*continuous.x,uv.y);
  float sideLine1=Box2(t1b-vec2(.5*continuous.y,3.5),vec2(.5*continuous.y,.1),.03),sideLine2=Box2(t1b-vec2(.5*continuous.y,-3.5),vec2(.5*continuous.y,.1),.03),separationLine1=Box2(params-vec2(.5*t3b.y,0),vec2(.5*t3b.y,.1),.01);
  return mix(vec3(1.-smoothstep(-length(fwidth(uv)),0.,min(min(sideLine1,sideLine2),separationLine1))),vec3(fract(uv),params.x),0.);
}
float smoothTerrainHeight(vec2 p)
{
  return 1e2*fBm(p*2./2e3,3,.6,.5);
}
vec2 roadSideItems(vec4 splineUV,float relativeHeight)
{
  vec3 pRoad=vec3(abs(splineUV.x),relativeHeight,splineUV.z),pObj=vec3(pRoad.x-4.2,pRoad.y-.8,0);
  relativeHeight=Box3(pObj,vec3(.1,.2,.1),.05);
  pObj=vec3(pRoad.x-4.1,pRoad.y-.8,0);
  relativeHeight=max(relativeHeight,-Box3(pObj,vec3(.1),.1));
  pObj=vec3(pRoad.x-4.3,pRoad.y-.5,round(pRoad.z*.5)/.5-pRoad.z);
  relativeHeight=min(relativeHeight,Box3(pObj,vec3(.05,.5,.05),.01));
  float reflector=Box3(pObj-vec3(-.1,.3,0),vec3(.04,.06,.03),.01);
  vec2 res=MinDist(vec2(relativeHeight,6),vec2(reflector,10));
  pObj=vec3(pRoad.x-4.5,pRoad.y-1.5,round(pRoad.z/30.)*30.-pRoad.z);
  relativeHeight=Box3(pObj,vec3(.1,3,.1),.1);
  res=MinDist(res,vec2(relativeHeight,6));
  pObj=vec3(pRoad.x-4.3,pRoad.y-4.,pObj.z);
  relativeHeight=Box3(pObj,vec3(.2,.1,.1),.1);
  return MinDist(res,vec2(relativeHeight,3));
}
vec2 terrainShape(vec3 p,vec4 splineUV)
{
  float terrainHeight=smoothTerrainHeight(p.xz),relativeHeight=p.y-terrainHeight;
  if(relativeHeight>5.5)
    return vec2(.75*relativeHeight,0);
  float isRoad=1.-smoothstep(roadWidthInMeters.x,roadWidthInMeters.y,abs(splineUV.x));
  if(isRoad<1.)
    terrainHeight+=.5*fBm(p.xz*2./1e2,1,.6,.5);
  float roadHeight=terrainHeight;
  if(isRoad>0.)
    {
      roadHeight=smoothTerrainHeight(GetPositionOnSpline(splineUV.yw));
      float x=clamp(abs(splineUV.x/roadWidthInMeters.x),0.,1.);
      roadHeight+=.2*(1.-x*x*x);
    }
  isRoad=mix(terrainHeight,roadHeight,isRoad);
  relativeHeight=p.y-isRoad;
  return MinDist(vec2(.75*relativeHeight,0),roadSideItems(splineUV,p.y-roadHeight));
}
float tree(vec3 globalP,vec3 localP,vec2 id,vec4 splineUV,float current_t)
{
  float h1=hash21(id),h2=fract(sin(h1)*43758.5453),presence=smoothstep(-.7,.7,fBm(id/5e2,2,.5,.3));
  if(h1<presence)
    return 1e6;
  if(abs(splineUV.x)<roadWidthInMeters.y)
    return 1e6;
  presence=mix(5.,20.,1.-h1*h1);
  float treeWidth=presence*mix(.3,.5,h2*h2),terrainHeight=smoothTerrainHeight(id);
  localP.y-=terrainHeight+.5*presence;
  localP.xz+=(vec2(h1,h2)*2.-1.)*2.;
  treeWidth=Ellipsoid(localP,.5*vec3(treeWidth,presence,treeWidth));
  terrainHeight=1.-smoothstep(50.,2e2,current_t);
  if(treeWidth<2.&&terrainHeight>0.)
    treeWidth+=terrainHeight*fBm(5.*vec2(2.*atan(localP.z,localP.x),localP.y)+id,2,.5,.5)*.5;
  return treeWidth;
}
vec2 treesShape(vec3 p,vec4 splineUV,float current_t)
{
  vec2 id=round(p.xz/10.)*10.;
  vec3 localP=p;
  localP.xz-=id;
  return vec2(tree(p,localP,id,splineUV,current_t),0);
}
vec3 motoPos,motoDir,headLightOffsetFromMotoRoot=vec3(.53,.98,0),breakLightOffsetFromMotoRoot=vec3(-1.14,.55,0),dirHeadLight=normalize(vec3(1,-.22,0)),dirBreakLight=normalize(vec3(-1,-.5,0));
vec3 motoToWorld(vec3 v,bool isPos,float time)
{
  float angle=atan(motoDir.z,motoDir.x);
  v.xz*=Rotation(-angle);
  if(isPos)
    v+=motoPos,v.z+=2.+.5*sin(time);
  return v;
}
vec3 worldToMoto(vec3 v,bool isPos,float time)
{
  if(isPos)
    v-=motoPos,v.z-=2.+.5*sin(time);
  time=atan(motoDir.z,motoDir.x);
  v.xz*=Rotation(time);
  return v;
}
vec3 meter3(vec2 uv,float value)
{
  float verticalLength=.04+.15*smoothstep(.1,.4,uv.x);
  vec3 baseCol=mix(vec3(.7,.9,.8),vec3(.8,0,0),smoothstep(.4,.41,uv.x));
  value*=.5;
  baseCol=smoothstep(.5,.7,fract(uv.x*30.))*smoothstep(.1,.3,fract(uv.y/verticalLength*2.))*mix(vec3(.01),baseCol,.15+.85*smoothstep(0.,.001,value-uv.x));
  return smoothstep(.001,0.,Box2(uv,vec2(.5,verticalLength),.01))*float(uv.y>0.)*baseCol;
}
vec3 meter4(vec2 uv)
{
  float value=.4,angle=atan(uv.y,uv.x);
  value=(value*1.5-1.)*PI;
  vec2 point=vec2(sin(value),cos(value))*.07;
  value=smoothstep(.004,.002,Segment3(uv.xyy,vec3(0),point.xyy,value));
  return smoothstep(.7,1.,mod(angle,.25)/.25)*smoothstep(0.,.01,abs(angle+.7)-.7)*smoothstep(0.,.01,.1-length(uv))*smoothstep(0.,.01,length(uv)-.06)*vec3(.36,.16,.12)+vec3(.7)*value;
}
float digit(int n,vec2 p)
{
  vec2 size=vec2(.2,.35);
  bool A=n!=1&&n!=4;
  p.x-=p.y*.15;
  float boundingBox=Box2(p,size,size.x*.5),innerBox=-Box2(p,size-.065,size.x*.15),d=1e6;
  if(A)
    {
      float sA=max(max(innerBox,.0125+p.x-p.y-size.x+size.y),.0125-p.x-p.y-size.x+size.y);
      d=min(d,sA);
    }
  if(n!=5&&n!=6)
    {
      float sB=max(max(max(innerBox,.0125-p.x+p.y+size.x-size.y),.0125-p.x-p.y-size.x+size.y),p.x-p.y-(size.x+.065)/2.);
      d=min(d,sB);
    }
  if(n!=2)
    {
      float sC=max(max(max(innerBox,.0125-p.x+p.y-size.x+size.y),.0125-p.x-p.y+size.x-size.y),p.x+p.y-(size.x+.065)/2.);
      d=min(d,sC);
    }
  if(A&&n!=7)
    {
      float sD=max(max(innerBox,.0125-p.x+p.y-size.x+size.y),.0125+p.x+p.y-size.x+size.y);
      d=min(d,sD);
    }
  if(A&&n%2==0)
    {
      float sE=max(max(max(innerBox,.0125+p.x-p.y+size.x-size.y),.0125+p.x+p.y-size.x+size.y),-p.x+p.y-(size.x+.065)/2.);
      d=min(d,sE);
    }
  if(n!=1&&n!=2&&n!=3&&n!=7)
    {
      float sF=max(max(max(innerBox,.0125+p.x-p.y-size.x+size.y),.0125+p.x+p.y+size.x-size.y),-p.x-p.y-(size.x+.065)/2.);
      d=min(d,sF);
    }
  if(n>1&&n!=7)
    {
      float sG=max(max(max(max(-.065+abs(p.y)*2.,.0125+p.x-p.y+size.x-size.y),.0125-p.x+p.y+size.x-size.y),.0125+p.x+p.y+size.x-size.y),.0125-p.x-p.y+size.x-size.y);
      d=min(d,sG);
    }
  return max(d,boundingBox);
}
vec3 glowy(float d)
{
  float dd=fwidth(d);
  vec3 segment=vec3(.67,.9,.8)*.5;
  return mix(mix(segment,vec3(.2),1.-1./exp(50.*max(0.,-d))),mix(vec3(0),segment,1./exp(2e2*max(0.,d))),smoothstep(-dd,dd,d));
}
vec3 motoDashboard(vec2 uv)
{
  vec3 color=meter3(uv*.6-vec2(.09,.05),.7+.3*sin(iTime*.5));
  color+=meter4(uv*.7-vec2(.6,.45));
  int speed=105+int(sin(iTime*.5)*10.);
  {
    vec2 uvSpeed=uv*3.-vec2(.4,1.95);
    if(speed>=100)
      color+=glowy(digit(speed/100,uvSpeed));
    color=color+glowy(digit(speed/10%10,uvSpeed-vec2(.5,0)))+glowy(digit(speed%10,uvSpeed-vec2(1,0)));
  }
  return color+glowy(digit(5,uv*8.-vec2(.7,2.4)));
}
material motoMaterial(float mid,vec3 p,vec3 N,float time)
{
  if(mid==2.)
    {
      vec3 luminance=smoothstep(.9,.95,N.x)*vec3(1,.95,.9);
      float isDashboard=smoothstep(.9,.95,-N.x+.4*N.y-.07);
      if(isDashboard>0.)
        {
          vec3 color=motoDashboard(p.zy*5.5+vec2(.5,-5));
          luminance=mix(vec3(0),color,isDashboard);
        }
      return material(2,luminance,.15);
    }
  return mid==3.?
    material(2,smoothstep(.9,.95,-N.x)*vec3(1,0,0),.5):
    mid==6.?
      material(1,vec3(1),.2):
      mid==5.?
        material(0,vec3(0),.3):
        mid==4.?
          material(0,vec3(.008),.8):
          mid==7.?
            material(0,vec3(.02,.025,.04),.6):
            mid==8.?
              material(0,vec3(0),.25):
              material(0,vec3(0),.15);
}
vec2 driverShape(vec3 p)
{
  p=worldToMoto(p,true,iTime)-vec3(-.35,.78,0);
  float d=length(p);
  if(d>1.2)
    return vec2(d,7);
  vec3 simP=p;
  simP.z=abs(simP.z);
  float wind=fBm((p.xy+iTime)*12.,1,.5,.5);
  if(d<.8)
    {
      vec3 pBody=simP;
      pBody.z-=.02;
      pBody.xy*=Rotation(3.1);
      pBody.yz*=Rotation(-.1);
      d=smin(d,Capsule(pBody,.12,.12),.1);
      pBody.y+=.2;
      pBody.xy*=Rotation(-.6);
      d=smin(d,Capsule(pBody,.12,.11),.02);
      pBody.y+=.2;
      pBody.xy*=Rotation(-.3);
      pBody.yz*=Rotation(-.2);
      d=smin(d,Capsule(pBody,.12,.12),.02);
      pBody.y+=.1;
      pBody.yz*=Rotation(1.7);
      d=smin(d,Capsule(pBody,.12,.1),.015);
    }
  d+=.005*wind;
  {
    vec3 pArm=simP-vec3(.23,.45,.18);
    pArm.yz*=Rotation(-.6);
    pArm.xy*=Rotation(.2);
    float arms=Capsule(pArm,.29,.06);
    d=smin(d,arms,.005);
    pArm.y+=.32;
    pArm.xy*=Rotation(1.5);
    arms=Capsule(pArm,.28,.04);
    d=smin(d,arms,.005);
  }
  d+=.01*wind;
  {
    vec3 pLeg=simP-vec3(0,0,.13);
    pLeg.xy*=Rotation(1.55);
    pLeg.yz*=Rotation(-.45);
    float h2=Capsule(pLeg,.35,.09);
    d=smin(d,h2,.01);
    pLeg.y+=.4;
    pLeg.xy*=Rotation(-1.5);
    h2=Capsule(pLeg,.4,.06);
    d=smin(d,h2,.01);
    pLeg.y+=.45;
    pLeg.xy*=Rotation(1.75);
    pLeg.yz*=Rotation(.25);
    h2=Capsule(pLeg,.2,.04);
    d=smin(d,h2,.01);
  }
  d+=.002*wind;
  {
    vec3 pHead=p-vec3(.39,.6,0);
    float head=length(pHead)-.15;
    if(head<d)
      return vec2(head,8);
  }
  return vec2(d,7);
}
vec2 motoShape(vec3 p)
{
  p=worldToMoto(p,true,iTime);
  float boundingSphere=length(p);
  if(boundingSphere>2.)
    return vec2(boundingSphere-1.5,1);
  vec2 d=vec2(1e6,1);
  float cyl;
  vec3 frontWheelPos=vec3(.9,.33,0);
  {
    vec3 pFrontWheel=p-frontWheelPos;
    float frontWheel=Torus(pFrontWheel.yzx,vec2(.26,.07));
    if(frontWheel<.25)
      {
        pFrontWheel.z=abs(pFrontWheel.z);
        cyl=Segment3(pFrontWheel,vec3(0,0,-1),vec3(0,0,1),boundingSphere);
        float frontBreak=-min(min(min(-cyl+.15,-pFrontWheel.z+.05),pFrontWheel.z-.04),cyl-.08);
        frontWheel=min(frontWheel,frontBreak);
      }
    d=MinDist(d,vec2(frontWheel,4));
  }
  {
    vec3 pRearWheel=p-vec3(-.85,.32,0);
    float rearWheel=Torus(pRearWheel.yzx,vec2(.23,.09));
    if(rearWheel<.25)
      {
        pRearWheel.z=abs(pRearWheel.z);
        cyl=Segment3(pRearWheel,vec3(0,0,-1),vec3(0,0,1),boundingSphere);
        float rearBreak=-min(min(min(-cyl+.15,-pRearWheel.z+.05),pRearWheel.z-.04),cyl-.08);
        rearWheel=min(rearWheel,rearBreak);
      }
    d=MinDist(d,vec2(rearWheel,4));
    {
      vec3 pBreak=p-breakLightOffsetFromMotoRoot;
      d=MinDist(d,vec2(Box3(pBreak,vec3(.02,.025,.1),.02),3));
    }
  }
  {
    vec3 pFork=p,pForkTop=vec3(-.48,.66,0),pForkAngle=pForkTop+vec3(-.14,.04,.05);
    pFork.z=abs(pFork.z);
    pFork-=frontWheelPos+vec3(0,0,.12);
    float fork=Segment3(pFork,pForkTop,vec3(0),boundingSphere)-.025;
    fork=min(fork,Segment3(pFork,pForkTop,pForkAngle,boundingSphere)-.0175);
    float handle=Segment3(pFork,pForkAngle,pForkAngle+vec3(-.08,-.07,.3),boundingSphere);
    fork=min(fork,handle-mix(.035,.02,smoothstep(.25,.4,boundingSphere)));
    pFork=pFork-pForkAngle-vec3(0,.1,.15);
    pFork.xz*=Rotation(.2);
    pFork.xy*=Rotation(-.2);
    handle=pFork.x-.02;
    pFork.xz*=Rotation(.25);
    handle=-min(handle,-Ellipsoid(pFork,vec3(.04,.05,.08)));
    fork=min(fork,handle);
    d=MinDist(d,vec2(fork,1));
  }
  {
    vec3 pHead=p-headLightOffsetFromMotoRoot;
    float headBlock=Ellipsoid(pHead,vec3(.15,.2,.15));
    if(headBlock<.2)
      {
        vec3 pHeadTopBottom=pHead;
        pHeadTopBottom.xy*=Rotation(-.15);
        headBlock=-min(min(min(-headBlock,-Ellipsoid(pHeadTopBottom-vec3(-.2,-.05,0),vec3(.35,.16,.25))),-Ellipsoid(pHead-vec3(-.2,-.08,0),vec3(.35,.25,.13))),-Ellipsoid(pHead-vec3(-.1,-.05,0),vec3(.2,.2,.3)));
        pHead.xy*=Rotation(-.4);
        headBlock=-min(-headBlock,-Ellipsoid(pHead-vec3(.1,0,0),vec3(.2,.3,.4)));
      }
    d=MinDist(d,vec2(headBlock,2));
    headBlock=Box3(p-vec3(.4,.82,0),vec3(.04,.1,.08),.02);
    d=MinDist(d,vec2(headBlock,5));
  }
  {
    vec3 pTank=p-vec3(.1,.74,0),pTankR=pTank;
    pTankR.xy*=Rotation(.45);
    pTankR.x+=.05;
    float tank=Ellipsoid(pTankR,vec3(.35,.2,.42));
    if(tank<.1)
      {
        float tankCut=Ellipsoid(pTankR+vec3(0,.13,0),vec3(.5,.35,.22));
        tank=-min(min(-tank,-tankCut),-Ellipsoid(pTank-vec3(0,.3,0),vec3(.6,.35,.4)));
      }
    d=MinDist(d,vec2(tank,1));
  }
  {
    vec3 pMotor=p-vec3(-.08,.44,0),pMotorSkewd=pMotor;
    pMotorSkewd.x*=1.-pMotorSkewd.y*.4;
    pMotorSkewd.x+=pMotorSkewd.y*.1;
    float motorBlock=Box3(pMotorSkewd,vec3(.44,.29,.11),.02);
    if(motorBlock<.5)
      {
        vec3 pMotor1=pMotor-vec3(.27,.12,0),pMotor2=pMotor-vec3(0,.12,0);
        pMotor1.xy*=Rotation(-.35);
        pMotor2.xy*=Rotation(.35);
        motorBlock=min(min(motorBlock,Box3(pMotor1,vec3(.1,.12,.2),.04)),Box3(pMotor2,vec3(.1,.12,.2),.04));
        pMotor1=pMotor-vec3(-.15,-.12,-.125);
        pMotor1.xy*=Rotation(-.15);
        float gearBox=Segment3(pMotor1,vec3(.2,0,0),vec3(-.15,0,0),boundingSphere);
        gearBox-=mix(.08,.15,boundingSphere);
        pMotor1.x+=.13;
        float gearBoxCut=min(-pMotor1.z-.05,Box3(pMotor1,vec3(.16,.08,.1),.04));
        gearBox=-min(-gearBox,-gearBoxCut);
        motorBlock=min(motorBlock,gearBox);
        gearBoxCut=Segment3(pMotor-vec3(.24,-.13,0),vec3(0,0,.4),vec3(0,0,-.4),boundingSphere)-.02;
        motorBlock=min(motorBlock,gearBoxCut);
      }
    d=MinDist(d,vec2(motorBlock,5));
  }
  {
    vec3 pExhaust=p-vec3(0,0,.14);
    float exhaust=Segment3(pExhaust,vec3(.24,.25,0),vec3(-.7,.3,.05),boundingSphere);
    if(exhaust<.6)
      exhaust=-min(-exhaust+mix(.04,.08,mix(boundingSphere,smoothstep(.5,.7,boundingSphere),.5)),p.x-.7*p.y+.9),exhaust=min(exhaust,Segment3(pExhaust,vec3(.24,.25,0),vec3(.32,.55,-.02),boundingSphere)-.04),exhaust=min(exhaust,Segment3(pExhaust,vec3(.22,.32,-.02),vec3(-.4,.37,.02),boundingSphere)-.04);
    d=MinDist(d,vec2(exhaust,6));
  }
  {
    vec3 pSeat=p-vec3(-.44,.44,0);
    float seat=Ellipsoid(pSeat,vec3(.8,.4,.2)),seatRearCut=length(p+vec3(1.05,-.1,0))-.7;
    seat=max(seat,-seatRearCut);
    if(seat<.2)
      {
        vec3 pSaddle=pSeat-vec3(.35,.57,0);
        pSaddle.xy*=Rotation(.4);
        float seatSaddleCut=Ellipsoid(pSaddle,vec3(.5,.15,.6));
        seat=-smin(min(-seat,seatSaddleCut),seatSaddleCut,.02);
        pSaddle=pSeat+vec3(0,-.55,0);
        pSaddle.xy*=Rotation(.5);
        seatSaddleCut=Ellipsoid(pSaddle,vec3(.8,.4,.4));
        seat=-min(-seat,-seatSaddleCut);
      }
    d=MinDist(d,vec2(seat,1));
  }
  return d;
}
light lights[3];
material computeMaterial(float mid,vec3 p,vec3 N)
{
  if(mid==0.)
    {
      vec3 color=pow(vec3(67,81,70)/255.*1.5,vec3(2.2));
      vec4 splineUV=ToSplineLocalSpace(p.xz,roadWidthInMeters.z);
      float isRoad=1.-smoothstep(roadWidthInMeters.x,roadWidthInMeters.y,abs(splineUV.x));
      vec3 roadColor=vec3(0);
      if(isRoad>0.)
        roadColor=roadPattern(splineUV.zx);
      color=mix(color,roadColor,isRoad);
      return material(0,color,.5);
    }
  return IsMoto(mid)?
    p=worldToMoto(p,true,iTime),N=worldToMoto(N,false,iTime),motoMaterial(mid,p,N,iTime):
    mid==10.?
      material(3,vec3(1,.4,0),.2):
      material(0,fract(p.xyz),1.);
}
vec2 sceneSDF(vec3 p,float current_t)
{
  vec2 d=vec2(1e6,-1);
  vec4 splineUV=ToSplineLocalSpace(p.xz,roadWidthInMeters.z);
  d=MinDist(d,motoShape(p));
  if(camShowDriver>.5)
    d=MinDist(d,driverShape(p));
  d=MinDist(d,terrainShape(p,splineUV));
  return MinDist(d,treesShape(p,splineUV,current_t));
}
void setLights()
{
  lights[0]=light(moonDirection*1e3,moonDirection,moonLightColor,-2.,1e10,.02);
  vec3 posHeadLight=motoToWorld(headLightOffsetFromMotoRoot,true,iTime),posBreakLight=motoToWorld(breakLightOffsetFromMotoRoot,true,iTime);
  dirHeadLight=motoToWorld(dirHeadLight,false,iTime);
  dirBreakLight=motoToWorld(dirBreakLight,false,iTime);
  lights[1]=light(posHeadLight,dirHeadLight,vec3(1),.93,10.,20.);
  lights[2]=light(posBreakLight,dirBreakLight,vec3(1,0,0),.7,2.,.1);
}
vec3 evalNormal(vec3 p,float t)
{
  vec2 k=vec2(1,-1);
  return normalize(k.xyy*sceneSDF(p+k.xyy*.002,t).x+k.yyx*sceneSDF(p+k.yyx*.002,t).x+k.yxy*sceneSDF(p+k.yxy*.002,t).x+k.xxx*sceneSDF(p+k.xxx*.002,t).x);
}
vec2 rayMarchScene(vec3 ro,vec3 rd,out vec3 p)
{
  p=ro;
  float t=0.;
  vec2 d;
  for(int i=ZERO(iTime);i<200;++i)
    {
      d=sceneSDF(p,t);
      t+=d.x;
      p=ro+t*rd;
      float epsilon=t*PIXEL_ANGLE;
      if(d.x<epsilon)
        return vec2(t,d.y);
      if(t>=1e2)
        return vec2(t,-1);
    }
  return vec2(t,d.y);
}
vec3 evalRadiance(vec2 t,vec3 p,vec3 V,vec3 N)
{
  float mid=t.y;
  if(mid==9.)
    return mix(abs(N.y)>.8?
      cityLights(p.xz*2.):
      vec3(0),mix(vec3(0),vec3(.06,.04,.03),V.y),min(t.x*.001,1.));
  if(mid==-1.)
    return sky(-V);
  material m=computeMaterial(mid,p,N);
  vec3 emissive=vec3(0);
  if(m.type==2)
    emissive=m.color;
  vec3 albedo=vec3(0);
  if(m.type==0)
    albedo=m.color;
  vec3 f0=vec3(.04);
  if(m.type==1||m.type==3)
    f0=m.color;
  albedo=vec3(0)+emissive+nightHorizonLight*mix(1.,.1,N.y*N.y)*(N.x*.5+.5)*albedo;
  if(m.roughness<.25)
    {
      vec3 L=reflect(-V,N);
      float x=1.-dot(V,normalize(L+V));
      x*=x*x*x*x;
      albedo+=f0*sky(L);
    }
  for(int i=0;i<3;++i)
    albedo=lights[i].cosAngle==-1.?
      albedo+rodLightContribution(m,lights[i],p,N,V):
      albedo+coneLightContribution(m,lights[i],p,N,V);
  return mix(albedo,vec3(0,0,.005)+vec3(.01,.01,.02)*.1,1.-exp(-t.x*.01));
}
void mainImage(out vec4 fragColor,vec2 fragCoord)
{
  ComputeBezierSegmentsLengthAndAABB();
  float time=fract((iTime+hash31(vec3(fragCoord,.001*iTime))*.008)*.1);
  motoPos.xz=GetPositionOnCurve(time);
  motoPos.y=smoothTerrainHeight(motoPos.xz);
  vec3 nextPos;
  nextPos.xz=GetPositionOnCurve(time+.01);
  nextPos.y=smoothTerrainHeight(nextPos.xz);
  motoDir=normalize(nextPos-motoPos);
  setLights();
  vec3 rd,cameraPosition=camPos,cameraTarget=camTa;
  if(camMotoSpace>.5)
    cameraPosition=motoToWorld(camPos,true,iTime),cameraTarget=motoToWorld(camTa,true,iTime);
  setupCamera((fragCoord/iResolution.xy*2.-1.)*vec2(1,iResolution.y/iResolution.x),cameraPosition,cameraTarget,camProjectionRatio,camFishEye,nextPos,rd);
  vec2 t=rayMarchScene(nextPos,rd,cameraPosition);
  cameraTarget=evalNormal(cameraPosition,t.x);
  rd=evalRadiance(t,cameraPosition,-rd,cameraTarget);
  fragColor=vec4(pow(rd,vec3(1./2.2)),1);
}
void main()
{
  mainImage(fragColor,gl_FragCoord.xy);
}

// src\shaders\scene.vert#version 150

in vec4 a_position;
out vec3 sunDir,camPos,camTa;
out float camFoV,camProjectionRatio,camFishEye,camMotoSpace,camShowDriver;
out vec2 spline[13];
uniform float iTime;
float hash11(float x)
{
  return fract(sin(x)*43758.5453);
}
vec2 hash12(float x)
{
  x=hash11(x);
  return vec2(x,hash11(x));
}
mat2 Rotation(float angle)
{
  float c=cos(angle);
  angle=sin(angle);
  return mat2(c,angle,-angle,c);
}
vec2 valueNoise(float p)
{
  float p0=floor(p);
  p-=p0;
  p=p*p*(3.-2.*p);
  return mix(hash12(p0),hash12(p0+1.),p);
}
void GenerateSpline()
{
  vec2 direction=normalize(vec2(hash11(1.),hash11(2.))*2.-1.),point=vec2(0);
  for(int i=0;i<13;i++)
    {
      if(i%2==0)
        {
          spline[i]=point+20.*direction;
          continue;
        }
      float ha=hash11(1.+float(i)*3.);
      point+=direction*40.;
      direction*=Rotation(mix(-1.8,1.8,ha));
      spline[i]=point;
    }
}
float verticalBump()
{
  return valueNoise(6.*iTime).x;
}
void sideShotFront()
{
  vec2 p=vec2(.95,.5);
  p.x+=mix(-.5,1.,valueNoise(.5*iTime).y);
  p.y+=.05*verticalBump();
  camPos=vec3(p,2.8);
  camTa=vec3(p.x,p.y+.1,0);
  camProjectionRatio=2.;
}
void sideShotRear()
{
  vec2 p=vec2(-1,.5);
  p.x+=mix(-1.2,.5,valueNoise(.5*iTime).y);
  p.y+=.05*verticalBump();
  camPos=vec3(p,2.8);
  camTa=vec3(p.x,p.y+.1,0);
  camProjectionRatio=2.;
}
void fpsDashboardShot()
{
  camPos=vec3(.1,1.12,0);
  camPos.z+=mix(-.02,.02,valueNoise(.1*iTime).x);
  camPos.y+=.01*valueNoise(5.*iTime).y;
  camTa=vec3(5,1,0);
  camProjectionRatio=.6;
}
void dashBoardUnderTheShoulderShot(float t)
{
  float bump=.02*verticalBump();
  camPos=vec3(-.2-.6*t,.88+.35*t+bump,.42);
  camTa=vec3(.5,1.+.2*t+bump,.25);
  camProjectionRatio=1.5;
}
void frontWheelCloseUpShot()
{
  camPos=vec3(-.1,.5,.5);
  camTa=vec3(.9,.35,.2);
  vec2 vibration=.005*valueNoise(40.*iTime);
  vibration.x+=.02*verticalBump();
  camPos.yz+=vibration;
  camTa.yz+=vibration;
  camProjectionRatio=1.6;
  camShowDriver=0.;
}
void overTheHeadShot()
{
  camPos=vec3(-1.4,1.7,0);
  camTa=vec3(.05,1.45,0);
  float bump=.01*verticalBump();
  camPos.y+=bump;
  camTa.y+=bump;
  camProjectionRatio=2.;
}
void viewFromBehind(float t_in_shot)
{
  camTa=vec3(1,1,0);
  camPos=vec3(-2.-4.*t_in_shot,.5,sin(t_in_shot));
  camProjectionRatio=1.;
}
void faceView(float t_in_shot)
{
  camTa=vec3(1,1.5,0);
  camPos=vec3(1.+3.*t_in_shot,1.5,1);
  camProjectionRatio=1.;
}
void main()
{
  gl_Position=a_position;
  GenerateSpline();
  camProjectionRatio=1.;
  camFishEye=.1;
  camMotoSpace=1.;
  camShowDriver=1.;
  float t=fract(iTime/6./8.)*8.,shot=floor(t);
  t=fract(t);
  if(shot==0.)
    sideShotRear();
  if(shot==1.)
    sideShotFront();
  if(shot==2.)
    frontWheelCloseUpShot();
  if(shot==3.)
    overTheHeadShot();
  if(shot==4.)
    fpsDashboardShot();
  if(shot==5.)
    dashBoardUnderTheShoulderShot(t);
  if(shot==6.)
    viewFromBehind(t);
  if(shot==7.)
    faceView(t);
  camFoV=atan(1./camProjectionRatio);
}

// src\shaders\fxaa.frag#version 150

out vec4 fragColor;
const vec2 iResolution=vec2(1920,1080);
uniform sampler2D tex;
void main()
{
  vec2 rcpFrame=1./iResolution,texcoord=gl_FragCoord.xy*rcpFrame;
  vec4 uv=vec4(texcoord,texcoord-rcpFrame*.5);
  vec3 luma=vec3(.299,.587,.114);
  float lumaNW=dot(textureLod(tex,uv.zw,0.).xyz,luma),lumaNE=dot(textureLod(tex,uv.zw+vec2(1,0)*rcpFrame.xy,0.).xyz,luma),lumaSW=dot(textureLod(tex,uv.zw+vec2(0,1)*rcpFrame.xy,0.).xyz,luma),lumaSE=dot(textureLod(tex,uv.zw+vec2(1)*rcpFrame.xy,0.).xyz,luma),lumaM=dot(textureLod(tex,uv.xy,0.).xyz,luma),lumaMin=min(lumaM,min(min(lumaNW,lumaNE),min(lumaSW,lumaSE)));
  lumaM=max(lumaM,max(max(lumaNW,lumaNE),max(lumaSW,lumaSE)));
  texcoord=vec2(-lumaNW-lumaNE+lumaSW+lumaSE,lumaNW+lumaSW-lumaNE-lumaSE);
  lumaNW=1./(min(abs(texcoord.x),abs(texcoord.y))+1./128.);
  texcoord=min(vec2(8),max(vec2(-8),texcoord*lumaNW))*rcpFrame.xy;
  vec3 rgbA=.5*(textureLod(tex,uv.xy+texcoord*(1./3.-.5),0.).xyz+textureLod(tex,uv.xy+texcoord*(2./3.-.5),0.).xyz),rgbB=rgbA*.5+.25*(textureLod(tex,uv.xy+texcoord*-.5,0.).xyz+textureLod(tex,uv.xy+texcoord*.5,0.).xyz);
  lumaNW=dot(rgbB,luma);
  fragColor=lumaNW<lumaMin||lumaNW>lumaM?
    vec4(rgbA,1):
    vec4(rgbB,1);
}

// src\shaders\postprocess.frag#version 150

out vec4 fragColor;
const vec2 iResolution=vec2(1920,1080);
uniform sampler2D tex;
void main()
{
  vec2 rcpFrame=1./iResolution,texcoord=gl_FragCoord.xy*rcpFrame;
  fragColor=texture(tex,vec4(texcoord,texcoord-rcpFrame*.5).xy);
}

