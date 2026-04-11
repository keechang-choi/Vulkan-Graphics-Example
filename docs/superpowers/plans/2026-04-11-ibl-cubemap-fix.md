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

## ibl 관련
ibl 계산하는 과정에서 pbr이 자연스럽지 않고 중간중간 격자형태 등으로 끊어지는 현상이 보임. 
이건 cubemap을 먼저 고친 후 점검 필요.
