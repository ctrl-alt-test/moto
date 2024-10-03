// src\shaders\preprocessed.scene.frag#version 150

vec2 iResolution=vec2(1920,1080);
uniform float iTime;
uniform sampler2D tex;
float camFishEye,camFoV,camMotoSpace,camProjectionRatio,camShowDriver;
vec3 camPos,camTa;
vec2 spline[13];
out vec4 fragColor;
float PIXEL_ANGLE=camFoV/iResolution.x,time;
const float PI=acos(-1.);
struct L{vec3 P;vec3 Q;vec3 C;float A;float B;float F;float I;};
struct M{int T;vec3 C;float R;};
float invV1(float NdotV,float sqrAlpha)
{
  return NdotV+sqrt(sqrAlpha+(1.-sqrAlpha)*NdotV*NdotV);
}
vec3 cookTorrance(vec3 f0,float roughness,vec3 NcrossH,float VdotH,float NdotL,float NdotV)
{
  roughness*=roughness;
  roughness*=roughness;
  float distribution=dot(NcrossH,NcrossH)*(1.-roughness)+roughness;
  VdotH=1.-VdotH;
  return(VdotH+f0*(1.-VdotH*VdotH*VdotH*VdotH*VdotH))*(roughness/(PI*distribution*distribution))*(1./(invV1(NdotV,roughness)*invV1(NdotL,roughness)));
}
vec3 lightContribution(L l,vec3 p,vec3 V,vec3 N,vec3 albedo,vec3 f0,float roughness)
{
  vec3 L,irradiance,L0=l.P-p;
  float d0=length(L0),NdotL;
  if(l.A!=-1.)
    {
      L=L0/d0;
      NdotL=dot(N,L);
      float LdotD=dot(L,-l.Q),angleFallOff=smoothstep(l.A,l.B,LdotD);
      angleFallOff*=angleFallOff;
      angleFallOff*=angleFallOff;
      angleFallOff*=angleFallOff;
      vec3 radiant_intensity=l.C*l.I*((sin((180.+vec3(0,.4,.8))/LdotD)*.5+.5)*.2+.8)*angleFallOff*((1.+l.F)/(l.F+d0*d0));
      irradiance=radiant_intensity*NdotL;
    }
  else
    {
      vec3 L1=l.Q-p;
      float d1=length(L1);
      d1=max(0.,dot(N,L0)/d0+dot(N,L1)/d1)/(d0*d1+dot(L0,L1));
      if(d1<=0.)
        return vec3(0);
      irradiance=l.C*l.I*d1;
      vec3 Ld=l.Q-l.P,R=reflect(-V,N);
      d1=dot(R,Ld);
      L=normalize(mix(L0,L1,clamp((dot(R,L0)*d1-dot(L0,Ld))/(dot(Ld,Ld)-d1*d1),0.,1.)));
      NdotL=dot(N,L);
    }
  if(NdotL<=0.)
    return vec3(0);
  L=normalize(L+V);
  L=cookTorrance(f0,roughness,cross(N,L),max(0.,dot(V,L)),NdotL,max(0.,dot(N,V)));
  return irradiance*(albedo+L);
}
float hash11(float x)
{
  return fract(sin(x)*43758.5453);
}
float hash21(vec2 xy)
{
  return fract(sin(dot(xy,vec2(12.9898,78.233)))*43758.5453);
}
float hash31(vec3 xyz)
{
  return hash21(vec2(hash21(xyz.xy),xyz.z));
}
vec2 hash12(float x)
{
  x=hash11(x);
  return vec2(x,hash11(x));
}
float valueNoise(vec2 p)
{
  vec2 p00=floor(p);
  p-=p00;
  p=p*p*(3.-2.*p);
  return mix(mix(hash21(p00),hash21(p00+vec2(1,0)),p.x),mix(hash21(p00+vec2(0,1)),hash21(p00+vec2(1)),p.x),p.y);
}
vec2 valueNoise2(float p)
{
  float p0=floor(p);
  p-=p0;
  p=p*p*(3.-2.*p);
  return mix(hash12(p0),hash12(p0+1.),p);
}
float fBm(vec2 p,int iterations,float weight_param,float frequency_param)
{
  float v=0.,weight=1.,frequency=1.,offset=0.;
  for(int i=min(0,int(iTime));i<iterations;++i)
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
  k/=1.-sqrt(.5);
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
vec4 FindCubicRoots(float a,float b,float c)
{
  float p=b-a*a/3.,p3=p*p*p;
  c+=a*(2.*a*a-9.*b)/27.;
  b=c*c+4.*p3/27.;
  a=-a/3.;
  if(b>=0.)
    {
      float z=sqrt(b);
      vec2 x=(vec2(z,-z)-c)/2.;
      x=sign(x)*pow(abs(x),vec2(1)/3.);
      return vec4(a+x.x+x.y,0,0,1);
    }
  p3=acos(-sqrt(-27./p3)*c/2.)/3.;
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
  vec4 res=vec4(min(A,C),max(A,C));
  if(B.x<res.x||B.x>res.z||B.y<res.y||B.y>res.w)
    {
      vec2 t=clamp((A-B)/(A-2.*B+C),0.,1.),s=1.-t;
      t=s*s*A+2.*s*t*B+t*t*C;
      res.xy=min(res.xy,t);
      res.zw=max(res.zw,t);
    }
  return res;
}
vec2 MinDist(vec2 d1,vec2 d2)
{
  return d1.x<d2.x?
    d1:
    d2;
}
void setupCamera(vec2 uv,vec3 cameraPosition,vec3 cameraTarget,out vec3 ro,out vec3 rd)
{
  vec3 cameraUp=vec3(0,1,0);
  cameraTarget=normalize(cameraTarget-cameraPosition);
  if(abs(dot(cameraTarget,cameraUp))>.99)
    cameraUp=vec3(1,0,0);
  vec3 cameraRight=normalize(cross(cameraTarget,cameraUp));
  cameraUp=normalize(cross(cameraRight,cameraTarget));
  uv*=mix(1.,length(uv),camFishEye);
  ro=cameraPosition;
  rd=normalize(cameraTarget*camProjectionRatio+uv.x*cameraRight+uv.y*cameraUp);
}
vec3 nightHorizonLight=.01*vec3(.07,.1,1),moonLightColor=vec3(.2,.8,1),moonDirection=normalize(vec3(-1,.3,.4));
vec3 sky(vec3 V)
{
  vec3 clearest=vec3(3,7,19);
  float direction=clamp(dot(V,normalize(vec3(0,1,.25))),0.,1.);
  vec3 color=mix(vec3(1,4,7),mix(vec3(10,15,13),mix(vec3(3,12,18),mix(clearest,vec3(0,0,1),pow(direction,.5)),smoothstep(0.,.3,mix(V.y,direction,.5))),smoothstep(0.,.15,mix(V.y,direction,.2))),smoothstep(.05,.15,V.y));
  direction=smoothstep(0.,1e-5,dot(V,moonDirection)-.9999);
  if(direction>0.)
    color=mix(color,clearest*5.*(smoothstep(-.5,.5,fBm(V.xy*1e2,4,.6,.7))+1.),direction);
  return color*mix(.1,1.,pow(smoothstep(0.,1.,fBm(.015*time+V.xz/(.01+V.y)*.5,5,.55,.7)+1.),2.))*smoothstep(-.1,0.,V.y)/2e3;
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
vec2 splineSegmentDistances[6];
void ComputeBezierSegmentsLengthAndAABB()
{
  float splineLength=0.;
  splineAABB=vec4(1e6,1e6,-1e6,-1e6);
  for(int i=min(0,int(iTime));i<6;++i)
    {
      int index=2*i;
      vec2 A=spline[index],B=spline[index+1],C=spline[index+2];
      splineSegmentDistances[i].x=splineLength;
      splineLength+=BezierCurveLengthAt(A,B,C,1.);
      splineSegmentDistances[i].y=splineLength;
      vec4 AABB=BezierAABB(A,B,C);
      splineSegmentAABBs[i]=AABB;
      splineAABB.xy=min(splineAABB.xy,AABB.xy);
      splineAABB.zw=max(splineAABB.zw,AABB.zw);
    }
}
vec4 ToSplineLocalSpace(vec2 p,float splineWidth)
{
  vec4 splineUV=vec4(1e6,0,0,0);
  if(DistanceFromAABB(p,splineAABB)>splineWidth)
    return splineUV;
  for(int i=min(0,int(iTime));i<6;++i)
    {
      int index=2*i;
      vec2 A=spline[index],B=spline[index+1],C=spline[index+2];
      if(DistanceFromAABB(p,BezierAABB(A,B,C))>splineWidth)
        continue;
      B=mix(B+vec2(1e-4),B,abs(sign(B*2.-A-C)));
      vec2 bezierSDF=BezierSDF(A,B,C,p);
      if(abs(bezierSDF.x)<abs(splineUV.x))
        {
          float lengthInSegment=BezierCurveLengthAt(A,B,C,clamp(bezierSDF.y,0.,1.))+splineSegmentDistances[i].x;
          splineUV=vec4(bezierSDF.x,clamp(bezierSDF.y,0.,1.),lengthInSegment,float(index));
        }
    }
  return splineUV;
}
vec2 GetPositionOnSpline(vec2 spline_t_and_index,out vec3 directionAndCurvature)
{
  float t=spline_t_and_index.x;
  int index=int(spline_t_and_index.y);
  spline_t_and_index=spline[index];
  vec2 B=spline[index+1],C=spline[index+2],AB=mix(spline_t_and_index,B,t),BC=mix(B,C,t);
  directionAndCurvature.xy=2.*(BC-AB);
  B=2.*(C-2.*B+spline_t_and_index);
  float norm=length(directionAndCurvature.xy);
  directionAndCurvature.z=directionAndCurvature.x*B.y-directionAndCurvature.y*B.x;
  directionAndCurvature.z/=norm*norm*norm;
  return mix(AB,BC,t);
}
vec2 GetTAndIndex(float t)
{
  t*=splineSegmentDistances[5].y;
  int index=0;
  for(;index<6&&t>splineSegmentDistances[index].y;++index)
    ;
  float segmentStartDistance=splineSegmentDistances[index].x,segmentEndDistance=splineSegmentDistances[index].y;
  return vec2((t-segmentStartDistance)/(segmentEndDistance-segmentStartDistance),index*2.);
}
vec3 roadWidthInMeters=vec3(4,8,8);
float roadMarkings(vec2 uv)
{
  vec2 params=vec2(.7,0),separationLineParams=vec2(13,3);
  if(params.x>.25)
    separationLineParams=vec2(6.5,1.5);
  if(params.x>.5)
    separationLineParams=vec2(26./6.,3);
  if(params.x>.75)
    separationLineParams=vec2(100);
  params=vec2(6.5,3);
  vec2 separationTileUV=vec2(fract(uv.x/separationLineParams.x)*separationLineParams.x,uv.y-floor(clamp(uv.y,0.,3.5)/3.5)*3.5),sideTileUV=vec2(fract((uv.x+.4)/params.x)*params.x,uv.y);
  float separationLine1=Box2(separationTileUV-vec2(.5*separationLineParams.y,0),vec2(.5*separationLineParams.y,.1),.01);
  return 1.-smoothstep(-.01,.01,min(min(Box2(sideTileUV-vec2(.5*params.y,3.5),vec2(.5*params.y,.1),.03),Box2(sideTileUV-vec2(.5*params.y,-3.5),vec2(.5*params.y,.1),.03)),separationLine1)+valueNoise(uv*30)*.03*valueNoise(uv));
}
M roadMaterial(vec2 uv)
{
  vec2 laneUV=uv/3.5;
  float tireTrails=sin((laneUV.x-.125)*4.*PI)*.5+.5;
  tireTrails=mix(mix(mix(tireTrails,smoothstep(0.,1.,tireTrails),.25),smoothstep(-.25,1.,fBm(laneUV*vec2(15,.1),2,.7,.4)),.2),fBm(laneUV*vec2(150,6),1,1.,1.),.1);
  float roughness=mix(.8,.4,tireTrails);
  vec3 color=vec3(mix(vec3(.11,.105,.1),vec3(.15),tireTrails));
  tireTrails=roadMarkings(uv.yx);
  color=mix(color,vec3(1),tireTrails);
  roughness=mix(roughness,.7,tireTrails);
  return M(tireTrails>.5?
    3:
    0,color,roughness);
}
float smoothTerrainHeight(vec2 p)
{
  return 50.*fBm(p*2./2e3,3,.6,.5);
}
float roadBumpHeight(float d)
{
  d=clamp(abs(d/roadWidthInMeters.x),0.,1.);
  return.2*(1.-d*d*d);
}
vec4 getRoadPositionDirectionAndCurvature(float t,out vec3 position)
{
  vec4 directionAndCurvature;
  position.xz=GetPositionOnSpline(GetTAndIndex(t),directionAndCurvature.xzw);
  position.y=smoothTerrainHeight(position.xz);
  directionAndCurvature.y=smoothTerrainHeight(position.xz+directionAndCurvature.xz)-position.y;
  directionAndCurvature.xyz=normalize(directionAndCurvature.xyz);
  return directionAndCurvature;
}
vec2 roadSideItems(vec4 splineUV,float relativeHeight)
{
  vec2 res=vec2(1e6,-1);
  vec3 pRoad=vec3(abs(splineUV.x),relativeHeight,splineUV.z);
  pRoad.x-=roadWidthInMeters.x*1.2;
  relativeHeight=smoothstep(0.,1.,abs(fract(splineUV.y*2.)*2.-1.)*2.)*2.-1.;
  vec3 pReflector=vec3(pRoad.x,pRoad.y-.8,round(pRoad.z/4.)*4.-pRoad.z);
  if(relativeHeight>-.5)
    {
      float height=.8*relativeHeight;
      vec3 pObj=vec3(pRoad.x,pRoad.y-height,0);
      float len=Box3(pObj,vec3(.1,.2,.1),.05);
      pObj=vec3(pRoad.x+.1,pRoad.y-height,0);
      len=max(len,-Box3(pObj,vec3(.1),.1));
      pObj=vec3(pRoad.x-.1,pRoad.y-height+.5,pReflector.z);
      len=min(len,Box3(pObj,vec3(.05,.5,.05),.01));
      res=MinDist(res,vec2(len,10));
    }
  if(relativeHeight>=1.||true)
    res=MinDist(res,vec2(Box3(pReflector,vec3(.05),.01),12));
  {
    vec3 pObj=vec3(pRoad.x-.7,pRoad.y,round(pRoad.z/50.)*50.-pRoad.z);
    float len=Box3(pObj,vec3(.1,7,.1),.1);
    pObj=vec3(pRoad.x+.7,pRoad.y-7.,pObj.z);
    pObj.xy*=Rotation(-.2);
    len=min(len,Box3(pObj,vec3(1.8,.05,.05),.1));
    res=MinDist(res,vec2(len,10));
    pObj.x+=1.2;
    len=Box3(pObj,vec3(.7,.1,.1),.1);
    res=MinDist(res,vec2(len,13));
  }
  {
    float len=max(-abs(pRoad.x+4)+4.8+max(pRoad.y-1,0)*.5,pRoad.y-4.);
    res=MinDist(res,vec2(min(len,max(len-.2,abs(mod(pRoad.z,10)-5)-.2)),11));
  }
  return res;
}
vec2 terrainShape(vec3 p,vec4 splineUV)
{
  float terrainHeight=smoothTerrainHeight(p.xz),relativeHeight=p.y-terrainHeight;
  if(relativeHeight>10.)
    return vec2(.75*relativeHeight,9);
  vec2 d=vec2(1e6,9);
  float isRoad=1.-smoothstep(roadWidthInMeters.x,roadWidthInMeters.y,abs(splineUV.x));
  if(isRoad<1.)
    terrainHeight+=valueNoise(p.xz*10.)*.1+.5*fBm(p.xz*2./1e2,1,.6,.5);
  float roadHeight=terrainHeight;
  if(isRoad>0.)
    {
      vec3 directionAndCurvature;
      vec2 positionOnSpline=GetPositionOnSpline(splineUV.yw,directionAndCurvature);
      roadHeight=smoothTerrainHeight(positionOnSpline);
      d=MinDist(d,roadSideItems(splineUV,p.y-roadHeight));
      roadHeight+=roadBumpHeight(splineUV.x)+pow(valueNoise(mod(p.xz*40,100)),.01)*.1;
    }
  roadHeight=mix(terrainHeight,roadHeight,isRoad);
  relativeHeight=p.y-roadHeight;
  return MinDist(d,vec2(.75*relativeHeight,9));
}
float tree(vec3 globalP,vec3 localP,vec2 id,vec4 splineUV,float current_t)
{
  float h1=hash21(id),h2=hash11(h1);
  if(h1<smoothstep(-.7,.7,fBm(id/5e2,2,.5,.3)))
    return 1e6;
  if(abs(splineUV.x)<roadWidthInMeters.y)
    return 1e6;
  float treeHeight=mix(5.,20.,1.-h1*h1),treeWidth=treeHeight*mix(.3,.5,h2*h2);
  localP.y-=smoothTerrainHeight(id)+.5*treeHeight;
  localP.xz+=(vec2(h1,h2)*2.-1.)*2.;
  treeHeight=Ellipsoid(localP,.5*vec3(treeWidth,treeHeight,treeWidth));
  treeWidth=1.-smoothstep(50.,2e2,current_t);
  if(treeHeight<2.&&treeWidth>0.)
    treeHeight+=treeWidth*fBm(5.*vec2(2.*atan(localP.z,localP.x),localP.y)+id,2,.5,.5)*.5;
  return treeHeight;
}
vec2 treesShape(vec3 p,vec4 splineUV,float current_t)
{
  vec2 id=round(p.xz/10.)*10.;
  vec3 localP=p;
  localP.xz-=id;
  return vec2(tree(p,localP,id,splineUV,current_t),9);
}
vec3 motoPos,headLightOffsetFromMotoRoot=vec3(.53,.98,0),breakLightOffsetFromMotoRoot=vec3(-1.14,.55,0);
float motoYaw,motoPitch,motoRoll,motoDistanceOnCurve;
void computeMotoPosition()
{
  motoDistanceOnCurve=mix(.1,.9,fract(time/20.));
  vec4 motoDirAndTurn=getRoadPositionDirectionAndCurvature(motoDistanceOnCurve,motoPos);
  float rightOffset=2.+.5*sin(time);
  motoPos.xz+=vec2(-motoDirAndTurn.z,motoDirAndTurn)*rightOffset;
  motoPos.y+=roadBumpHeight(abs(rightOffset))+.1;
  motoYaw=atan(motoDirAndTurn.z,motoDirAndTurn.x);
  motoPitch=atan(motoDirAndTurn.y,length(motoDirAndTurn.zx));
  motoRoll=20.*motoDirAndTurn.w;
}
vec3 motoToWorld(vec3 v,bool isPos)
{
  v.xy*=Rotation(-motoPitch);
  v.yz*=Rotation(-motoRoll);
  v.xz*=Rotation(-motoYaw);
  if(isPos)
    v+=motoPos;
  return v;
}
vec3 worldToMoto(vec3 v,bool isPos)
{
  if(isPos)
    v-=motoPos;
  v.xz*=Rotation(motoYaw);
  v.yz*=Rotation(motoRoll);
  v.xy*=Rotation(motoPitch);
  return v;
}
vec3 meter3(vec2 uv,float value)
{
  float verticalLength=.04+.15*smoothstep(.1,.4,uv.x);
  return smoothstep(.001,0.,Box2(uv,vec2(.5,verticalLength),.01))*float(uv.y>0.)*(smoothstep(.5,.7,fract(uv.x*30.))*smoothstep(.1,.3,fract(uv.y/verticalLength*2.))*mix(vec3(.01),mix(vec3(.7,.9,.8),vec3(.8,0,0),smoothstep(.4,.41,uv.x)),.15+.85*smoothstep(0.,.001,value*.5-uv.x)));
}
vec3 meter4(vec2 uv)
{
  float value=-.4*PI,angle=atan(uv.y,uv.x),dummy;
  value=smoothstep(.004,.002,Segment3(uv.xyy,vec3(0),(vec2(sin(value),cos(value))*.07).xyy,dummy));
  return smoothstep(.7,1.,mod(angle,.25)/.25)*smoothstep(0.,.01,abs(angle+.7)-.7)*smoothstep(0.,.01,.1-length(uv))*smoothstep(0.,.01,length(uv)-.06)*vec3(.36,.16,.12)+vec3(.7)*value;
}
float digit(int n,vec2 p2)
{
  vec2 size=vec2(.2,.35);
  bool A=n!=1&&n!=4;
  p2-=vec2(p2.y*.15,0);
  float innerBox=-Box2(p2,size-.06,0.),d=1e6;
  if(A)
    d=min(d,max(max(innerBox,.01+p2.x-p2.y-size.x+size.y),.01-p2.x-p2.y-size.x+size.y));
  if(n!=5&&n!=6)
    d=min(d,max(max(max(innerBox,.01-p2.x+p2.y+size.x-size.y),.01-p2.x-p2.y-size.x+size.y),p2.x-p2.y-(size.x+.06)/2.));
  if(n!=2)
    d=min(d,max(max(max(innerBox,.01-p2.x+p2.y-size.x+size.y),.01-p2.x-p2.y+size.x-size.y),p2.x+p2.y-(size.x+.06)/2.));
  if(A&&n!=7)
    d=min(d,max(max(innerBox,.01-p2.x+p2.y-size.x+size.y),.01+p2.x+p2.y-size.x+size.y));
  if(A&&n%2==0)
    d=min(d,max(max(max(innerBox,.01+p2.x-p2.y+size.x-size.y),.01+p2.x+p2.y-size.x+size.y),-p2.x+p2.y-(size.x+.06)/2.));
  if(n!=1&&n!=2&&n!=3&&n!=7)
    d=min(d,max(max(max(innerBox,.01+p2.x-p2.y-size.x+size.y),.01+p2.x+p2.y+size.x-size.y),-p2.x-p2.y-(size.x+.06)/2.));
  if(n>1&&n!=7)
    d=min(d,max(max(max(max(-.06+abs(p2.y)*2.,.01+p2.x-p2.y+size.x-size.y),.01-p2.x+p2.y+size.x-size.y),.01+p2.x+p2.y+size.x-size.y),.01-p2.x-p2.y+size.x-size.y));
  return max(d,Box2(p2,size,size.x*.5));
}
vec3 glowy(float d)
{
  return mix(mix(vec3(.2),vec3(.5),1./exp(50.*max(0.,-d))),mix(vec3(0),vec3(.5),1./exp(2e2*max(0.,d))),smoothstep(0.,.01,d));
}
vec3 motoDashboard(vec2 uv)
{
  int speed=105+int(sin(iTime*.5)*10.);
  vec2 uvSpeed=uv*3.-vec2(.4,1.95);
  return meter3(uv*.6-vec2(.09,.05),.7+.3*sin(time*.5))+meter4(uv*.7-vec2(.6,.45))+glowy(min(min(min(digit(5,uv*8.-vec2(.7,2.4)),float(speed<100)+digit(speed/100,uvSpeed)),digit(speed/10%10,uvSpeed-vec2(.5,0))),digit(speed%10,uvSpeed-vec2(1,0))));
}
M motoMaterial(int mid,vec3 p,vec3 N)
{
  if(mid==5)
    {
      vec3 luminance=smoothstep(0.,.01,N.x-.94)*vec3(1,.95,.9);
      float isDashboard=smoothstep(.9,.95,-N.x+.4*N.y-.07);
      if(isDashboard>0.)
        {
          vec3 color=motoDashboard(p.zy*5.5+vec2(.5,-5));
          luminance=mix(vec3(0),color,isDashboard);
        }
      return M(2,luminance,.08);
    }
  return mid==4?
    M(2,smoothstep(.9,.95,-N.x)*mix(vec3(1,.005,.02),vec3(.02,0,0),smoothstep(.2,1.,sqrt(length(fract(68.*p.yz+vec2(.6,0))*2.-1.)))),.5):
    mid==3?
      M(1,vec3(1),.05):
      mid==2?
        M(0,vec3(0),.3):
        mid==1?
          M(0,vec3(.008),.8):
          mid==6?
            M(0,vec3(.02,.025,.04),.6):
            mid==7?
              M(0,vec3(0),.12):
              M(0,vec3(0),.08);
}
vec2 driverShape(vec3 p)
{
  p=worldToMoto(p,true)-vec3(-.35,.78,0);
  float d=length(p);
  if(d>1.2||camShowDriver<.5)
    return vec2(d,6);
  vec3 simP=p;
  simP.z=abs(simP.z);
  float wind=fBm((p.xy+time)*12.,1,.5,.5);
  if(d<.8)
    {
      vec3 pBody=p;
      pBody.z=max(abs(pBody.z)-.02,0);
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
      pBody=p;
      pBody.y-=.48;
      pBody.x-=.25;
      pBody.xy*=Rotation(-.7);
      d=min(d,length(vec2(max(abs(pBody.y)-.07,0),abs(length(pBody.xz)-.05)))-.04);
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
    float head=max(length(pHead*vec3(1,1,1.2+pHead.y))-.15,-pHead.y-.09-pHead.x);
    if(head<d)
      return vec2(head,7);
  }
  return vec2(d,6);
}
vec2 wheelShape(vec3 p,float wheelRadius,float tireRadius,float innerRadius,vec3 innerPart)
{
  wheelRadius=Torus(p.yzx,vec2(wheelRadius,tireRadius));
  if(wheelRadius<.25)
    {
      p.z=abs(p.z);
      float h;
      h=Segment3(p,vec3(0),vec3(0,0,1),h);
      wheelRadius=min(min(-smin(-wheelRadius,h-innerRadius,.01),-min(min(min(.15-h,h-.08),p.z-.04),-p.z+.05)),Ellipsoid(p,innerPart));
    }
  return vec2(wheelRadius,1);
}
vec2 motoShape(vec3 p)
{
  p=worldToMoto(p,true);
  float boundingSphere=length(p);
  if(boundingSphere>2.)
    return vec2(boundingSphere-1.5,0);
  vec2 d=vec2(1e6,0);
  vec3 frontWheelPos=vec3(.9,.33,0);
  d=MinDist(d,wheelShape(p-frontWheelPos,.26,.07,.22,vec3(.02,.02,.12)));
  d=MinDist(d,wheelShape(p-vec3(-.85,.32,0),.17,.15,.18,vec3(.2,.2,.01)));
  {
    vec3 pBreak=p-breakLightOffsetFromMotoRoot;
    d=MinDist(d,vec2(Box3(pBreak,vec3(.02,.025,.1),.02),4));
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
    d=MinDist(d,vec2(fork,0));
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
    d=MinDist(d,vec2(headBlock,5));
    headBlock=Box3(p-vec3(.4,.82,0),vec3(.04,.1,.08),.02);
    d=MinDist(d,vec2(headBlock,2));
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
    d=MinDist(d,vec2(tank,0));
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
    d=MinDist(d,vec2(motorBlock,2));
  }
  {
    vec3 pExhaust=p-vec3(0,0,.2);
    float exhaust=Segment3(pExhaust,vec3(.24,.25,0),vec3(-.7,.3,.05),boundingSphere);
    if(exhaust<.6)
      exhaust=-min(-exhaust+mix(.04,.08,mix(boundingSphere,smoothstep(.5,.7,boundingSphere),.5)),p.x-.7*p.y+.9),exhaust=min(exhaust,Segment3(pExhaust,vec3(.24,.25,0),vec3(.32,.55,-.02),boundingSphere)-.04),exhaust=min(exhaust,Segment3(pExhaust,vec3(.22,.32,-.02),vec3(-.4,.37,.02),boundingSphere)-.04);
    d=MinDist(d,vec2(exhaust,0));
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
    d=MinDist(d,vec2(seat,0));
  }
  return d;
}
M computeMaterial(int mid,vec3 p,vec3 N)
{
  vec4 splineUV;
  vec3 pRoad=p;
  if(mid>=9)
    splineUV=ToSplineLocalSpace(p.xz,roadWidthInMeters.z),pRoad.xz=splineUV.xz;
  if(mid==9)
    {
      float isRoad=1.-smoothstep(roadWidthInMeters.x,roadWidthInMeters.y,abs(splineUV.x));
      vec3 grassColor=pow(vec3(67,81,70)/255.,vec3(2.2));
      if(isRoad>0.)
        {
          M m=roadMaterial(splineUV.xz);
          m.C=mix(grassColor,m.C,isRoad);
          return m;
        }
      return M(0,grassColor,.5);
    }
  if(mid<=7)
    return p=worldToMoto(p,true),N=worldToMoto(N,false),motoMaterial(mid,p,N);
  M utility=M(1,vec3(.9),.7);
  return mid==10?
    utility:
    mid==13?
      N.y>-.5?
        utility:
        M(2,vec3(5,3,.1),.4):
      mid==11?
        M(0,vec3(.5)+fBm(pRoad.yz*vec2(.2,1)+valueNoise(pRoad.xz),3,.6,.9)*.15,.6):
        mid==12?
          M(3,vec3(1,.4,.05),.2):
          M(0,fract(p.xyz),1.);
}
vec2 sceneSDF(vec3 p,float current_t)
{
  vec4 splineUV=ToSplineLocalSpace(p.xz,roadWidthInMeters.z);
  vec2 d=motoShape(p);
  d=MinDist(d,driverShape(p));
  d=MinDist(d,terrainShape(p,splineUV));
  return MinDist(d,treesShape(p,splineUV,current_t));
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
  for(int i=min(0,int(iTime));i<200;++i)
    {
      d=sceneSDF(p,t);
      t+=d.x;
      p=ro+t*rd;
      float epsilon=t*PIXEL_ANGLE;
      if(d.x<epsilon)
        return vec2(t,d.y);
      if(t>=1e3)
        return vec2(t,-1);
    }
  return vec2(t,d.y);
}
vec3 evalRadiance(vec2 t,vec3 p,vec3 V,vec3 N)
{
  int mid=int(t.y);
  if(mid==8)
    return mix(abs(N.y)>.8?
      cityLights(p.xz*2.):
      vec3(0),mix(vec3(0),vec3(.06,.04,.03),V.y),min(t.x*.001,1.));
  if(mid==-1)
    return sky(-V);
  M m=computeMaterial(mid,p,N);
  vec3 emissive=vec3(0);
  if(m.T==2)
    {
      float aligned=clamp(dot(V,N),0.,1.),aligned4=aligned*aligned*aligned*aligned;
      emissive=m.C*mix(aligned*.1+aligned4,1.,aligned4*aligned4*aligned4*aligned4*aligned4*aligned4*aligned4*aligned4);
    }
  vec3 albedo=vec3(0);
  if(m.T==0)
    albedo=m.C;
  vec3 f0=vec3(.04);
  if(m.T==1)
    f0=m.C;
  if(m.T==3)
    f0=m.C,N=V;
  vec3 I0=.01*vec3(.07,.1,1)*mix(1.,.1,N.y*N.y)*(N.x*.5+.5);
  emissive+=I0*albedo;
  if(m.R<.25)
    {
      vec3 L=reflect(-V,N);
      emissive+=f0*sky(L);
    }
  for(int i=0;i<19;++i)
    {
      L light;
      if(i==16)
        light=L(moonDirection*1e3,-moonDirection,vec3(.2,.8,1),0.,0.,1e10,.005);
      if(i==17)
        {
          vec3 pos=motoToWorld(headLightOffsetFromMotoRoot+vec3(.1,0,0),true),dir=motoToWorld(normalize(vec3(1,-.15,0)),false);
          light=L(pos,dir,vec3(1),.75,.95,10.,5.);
        }
      if(i==18)
        {
          vec3 pos=motoToWorld(breakLightOffsetFromMotoRoot,true),dir=motoToWorld(normalize(vec3(-1,-.5,0)),false);
          light=L(pos,dir,vec3(1,0,0),.3,.9,2.,.05);
        }
      if(i<16)
        {
          float t=float(i/2-4+1),roadLength=splineSegmentDistances[5].y,motoDistanceOnRoad=motoDistanceOnCurve*roadLength;
          t=(floor(motoDistanceOnRoad/50.)*50.+t*50.)/roadLength;
          if(t>=1.)
            continue;
          vec3 pos;
          vec4 roadDirAndCurve=getRoadPositionDirectionAndCurvature(t,pos);
          roadDirAndCurve.y=0.;
          pos.x+=(roadWidthInMeters.x-1.)*1.2*(float(i%2)*2.-1.);
          pos.y+=5.;
          light=L(pos,pos+roadDirAndCurve.xyz,vec3(1,.3,0),-1.,0.,0.,10.);
        }
      emissive+=lightContribution(light,p,V,N,albedo,f0,m.R);
    }
  emissive=mix(emissive,vec3(.001,.001,.005),1.-exp(-t.x*.01));
  return emissive*2;
}
void GenerateSpline()
{
  float seed=2.+floor(iTime/20);
  vec2 direction=normalize(vec2(hash11(seed),hash11(seed+1.))*2.-1.),point=vec2(0);
  for(int i=0;i<13;i++)
    {
      if(i%2==0)
        {
          spline[i]=point+40.*direction;
          continue;
        }
      float ha=hash11(seed+float(i)*3.);
      point+=direction*80.;
      direction*=Rotation(mix(-1.8,1.8,ha));
      spline[i]=point;
    }
}
float verticalBump()
{
  return valueNoise2(6.*iTime).x;
}
void sideShotFront()
{
  vec2 p=vec2(.95,.5);
  p.x+=mix(-.5,1.,valueNoise2(.5*iTime).y);
  p.y+=.05*verticalBump();
  camPos=vec3(p,1.5);
  camTa=vec3(p.x,p.y+.1,0);
  camProjectionRatio=1.2;
}
void sideShotRear()
{
  vec2 p=vec2(-1,.5);
  p.x+=mix(-.2,.2,valueNoise2(.5*iTime).y);
  p.y+=.05*verticalBump();
  camPos=vec3(p,1.5);
  camTa=vec3(p.x,p.y+.1,0);
  camProjectionRatio=1.2;
}
void fpsDashboardShot()
{
  camPos=vec3(.1,1.12,0);
  camPos.z+=mix(-.02,.02,valueNoise2(.1*iTime).x);
  camPos.y+=.01*valueNoise2(5.*iTime).y;
  camTa=vec3(5,1,0);
  camProjectionRatio=.7;
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
  vec2 vibration=.005*valueNoise2(40.*iTime);
  vibration.x+=.02*verticalBump();
  camPos.yz+=vibration;
  camTa.yz+=vibration;
  camProjectionRatio=1.6;
  camShowDriver=0.;
}
void overTheHeadShot()
{
  camPos=vec3(-1.8,1.7,0);
  camTa=vec3(.05,1.45,0);
  float bump=.01*verticalBump();
  camPos.y+=bump;
  camTa.y+=bump;
  camProjectionRatio=3.;
}
void viewFromBehind(float t_in_shot)
{
  camTa=vec3(1,1,0);
  camPos=vec3(-2.-2.5*t_in_shot,1.5,sin(t_in_shot));
  camProjectionRatio=1.;
}
void faceView(float t_in_shot)
{
  camTa=vec3(1,1.5,0);
  camPos=vec3(1.+3.*t_in_shot,1.5,1);
  camProjectionRatio=1.;
}
void openingShot(float t_in_shot)
{
  camTa=vec3(10,12.-mix(0.,10.,min(1.,t_in_shot/6.)),1);
  camPos=vec3(5,7.-min(t_in_shot,6.),1);
  camProjectionRatio=1.;
}
bool get_shot(inout float time,float duration)
{
  if(time<duration)
    return true;
  time-=duration;
  return false;
}
void selectShot()
{
  float time=iTime;
  GenerateSpline();
  camProjectionRatio=1.;
  camFishEye=.1;
  camMotoSpace=1.;
  camShowDriver=1.;
  camFoV=atan(1./camProjectionRatio);
  if(get_shot(time,10.5))
    openingShot(time);
  else if(get_shot(time,6.))
    sideShotRear();
  else if(get_shot(time,5.))
    sideShotFront();
  else if(get_shot(time,8.))
    frontWheelCloseUpShot();
  else if(get_shot(time,8.))
    overTheHeadShot();
  else if(get_shot(time,8.))
    fpsDashboardShot();
  else if(get_shot(time,8.))
    dashBoardUnderTheShoulderShot(time);
  else if(get_shot(time,8.))
    viewFromBehind(time);
  else if(get_shot(time,8.))
    faceView(time);
  else if(get_shot(time,8.))
    sideShotRear();
  else if(get_shot(time,8.))
    sideShotFront();
  else if(get_shot(time,8.))
    frontWheelCloseUpShot();
  else if(get_shot(time,8.))
    overTheHeadShot();
  else if(get_shot(time,8.))
    fpsDashboardShot();
  else if(get_shot(time,8.))
    dashBoardUnderTheShoulderShot(time);
  else if(get_shot(time,8.))
    viewFromBehind(time);
  else
     overTheHeadShot();
}
void main()
{
  ComputeBezierSegmentsLengthAndAABB();
  selectShot();
  vec2 texCoord=gl_FragCoord.xy/iResolution.xy,uv=(texCoord*2.-1.)*vec2(1,iResolution.y/iResolution.x);
  time=iTime+hash31(vec3(gl_FragCoord.xy,.001*iTime))*.008;
  computeMotoPosition();
  vec3 ro,rd,cameraPosition=camPos,cameraTarget=camTa;
  if(camMotoSpace>.5)
    cameraPosition=motoToWorld(camPos,true),cameraTarget=motoToWorld(camTa,true);
  setupCamera(uv,cameraPosition,cameraTarget,ro,rd);
  vec2 t=rayMarchScene(ro,rd,cameraPosition);
  fragColor=vec4(mix(pow(evalRadiance(t,cameraPosition,-rd,evalNormal(cameraPosition,t.x)),vec3(1./2.2))*smoothstep(0.,4.,iTime)*smoothstep(138.,132.,iTime),texture(tex,texCoord).xyz,.2)+vec3(hash21(fract(uv+iTime)),hash21(fract(uv-iTime)),hash21(fract(uv.yx+iTime)))*.025,1);
}

// src\shaders\preprocessed.fxaa.frag#version 150

out vec4 fragColor;
vec2 iResolution=vec2(1920,1080);
uniform sampler2D tex;
void main()
{
  vec2 rcpFrame=1./iResolution,texcoord=gl_FragCoord.xy*rcpFrame,uv=texcoord;
  texcoord-=rcpFrame*.5;
  vec4 luma=vec4(.299,.587,.114,0);
  float lumaNW=dot(texture(tex,texcoord),luma),lumaNE=dot(texture(tex,texcoord+vec2(1,0)*rcpFrame.xy),luma),lumaSW=dot(texture(tex,texcoord+vec2(0,1)*rcpFrame.xy),luma),lumaSE=dot(texture(tex,texcoord+rcpFrame.xy),luma),lumaM=dot(texture(tex,uv),luma);
  texcoord=vec2(-lumaNW-lumaNE+lumaSW+lumaSE,lumaNW+lumaSW-lumaNE-lumaSE);
  float rcpDirMin=1./(min(abs(texcoord.x),abs(texcoord.y))+1./128.);
  texcoord=min(vec2(8),max(vec2(-8),texcoord*rcpDirMin))*rcpFrame.xy;
  vec4 rgbA=.5*(texture(tex,uv+texcoord*(1./3.-.5))+texture(tex,uv+texcoord*(2./3.-.5))),rgbB=rgbA*.5+.25*(texture(tex,uv+texcoord*-.5)+texture(tex,uv+texcoord*.5));
  rcpDirMin=dot(rgbB,luma);
  fragColor=rcpDirMin<min(lumaM,min(min(lumaNW,lumaNE),min(lumaSW,lumaSE)))||rcpDirMin>max(lumaM,max(max(lumaNW,lumaNE),max(lumaSW,lumaSE)))?
    rgbA:
    rgbB;
}

// src\shaders\preprocessed.postprocess.frag#version 150

out vec4 fragColor;
vec2 iResolution=vec2(1920,1080);
uniform sampler2D tex;
void main()
{
  vec2 uv=gl_FragCoord.xy/iResolution;
  fragColor=1-(1-texture(tex,uv))*(1-textureLod(tex,uv,7))*(1-textureLod(tex,1-uv,5)*.2);
  fragColor*=max(1-pow(length(uv-.5),1.5)*1.5,0);
}

