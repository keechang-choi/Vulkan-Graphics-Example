# problems
일단 cubemap이 baking은 되는데, view matrix에 문제가 있다.  
아마도 onetime submit인데 ubo로 viewmatrix 업데이트 하면서 호출하니까 cpu 업데이트는 되는데 gpu로 전달이 안되는 것 같다. 이게 맞다면 capture mvp를 전부 push constants로 바꿔야할 것 같다.

**[수정 완료 - 2026-04-11]** `equirect.vert`의 CaptureUbo(set=0)를 push constant로 교체.
`buildEnvCubemap`, `buildIrradianceMap`, `buildPrefilteredMap` 모두 captureUboBuf/captureUboDS 제거하고
`cmd.pushConstants`로 face별 MVP를 커맨드 버퍼에 직접 기록. (commit: e987c67)

## depth 관련.
debug display viewport를 렌더할때, main viewport와 depth 가 처리되고 있음.
debug display 상의 object 일부가 main vieweport의 object에 의해 가려지는 경우가 발생함.
debug display viewport를 render할때는 depth가 항상 맨 앞에 오도록 순서를 수정하거나 depth 값 수정이 필요해보임.

**[수정 완료 - 2026-04-11]** `displayTargets` derivative 파이프라인 생성 시 `noDepthSCI(depthTest=false, depthWrite=false)`를
별도로 적용하여 debug view quad가 항상 scene 위에 렌더링되도록 수정. (commit: fc2934a)

## Done. 헬멧 개수와 grid 옵션 조절
1x1 ~ 8x8 변경할때도 모두 중앙에 위치되어야 해
그리고 그에따른 metallic과 roughness 조절도 grid 현재 수에 맞게 0~1 사이 interpolation해야함.

## ibl 관련
ibl 계산하는 과정에서 pbr이 자연스럽지 않고 중간중간 격자형태 등으로 끊어지는 현상이 보임. 
이건 cubemap을 먼저 고친 후 점검 필요.
더 확인해보니 모델의 종류 sphere, helmet 무관하게 항상  나타나고 무아레 무늬 같은 현상이 IBL을 키면 보임.
metallic이 1일때는 안보이고 0에 가까워질수록 이현상이 크게 나타남

지금까지 분석된 내용은 다음과 같다. 다음에 이어서 진행
❯ 어 보인다. 근데 이게 irradiance map baking에 문제가 있는거면 ibl만 켰을때 보이는 것도 설명이 되는거야?

● 맞아요! irradiance map이 baking 단계에서 이미 격자 패턴을 가지고 있으니, IBL을 켜면 그 패턴이 오브젝트에 그대로 찍히는
   거죠. IBL 꺼져있을 땐 irradiance map을 전혀 안 쓰니까 안 보이는 게 당연합니다.

  원인: irradiance.frag의 균일 phi-theta 격자 샘플링이 HDR 환경의 고주파 성분(밝은 하늘, 태양 등)과 공명해서 aliasing이
  생김. 고정된 world Y up 벡터 기반 tangent frame과 결합돼 규칙적인 격자로 나타납니다.

  LOD=0이 더 심해진 이유: 블러된 mip 대신 선명한 mip 0을 쓰니 aliasing이 더 크게 나타난 것입니다.

  지금의 Riemann sum 방식을 Hammersley quasi-random + cosine-weighted 중요도 샘플링으로 교체합니다.

