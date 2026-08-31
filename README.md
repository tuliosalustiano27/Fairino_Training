# Fairino FR5 — VLA Training Pipeline

Trilha de engenharia para treinar um modelo Vision-Language-Action (VLA) capaz
de executar uma tarefa de *pick* no robô real **Fairino FR5**, validada
antes em simulação (NVIDIA Isaac Sim). O projeto é didático por design: cada
decisão técnica é documentada com o porquê, não só o quê, para servir de
referência reproduzível.

| | |
|---|---|
| **Robô alvo** | Fairino FR5 (6 DOF) + garra Robotiq 2F-85 |
| **Simulador** | NVIDIA Isaac Sim 6.0.1 (remoto, via MCP server) |
| **Modelo VLA** | [SmolVLA](https://github.com/huggingface/lerobot) (450M) — principal |
| **Modelo alternativo** | [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) N1.7 (3B) — validado, reservado para produção |
| **Formato de dataset** | LeRobot v2 (Parquet + MP4) |
| **Hardware de treino** | RTX 5090 32GB (WSL2 Ubuntu 22.04) |
| **Status** | Em andamento — pipeline de gravação→dataset validado ponta a ponta; pega da garra ainda não confiável |

---

## Sumário

- [Arquitetura do pipeline](#arquitetura-do-pipeline)
- [Status atual](#status-atual)
- [Decisões técnicas](#decisões-técnicas)
- [Cinemática do FR5](#cinemática-do-fr5)
- [Infraestrutura](#infraestrutura)
- [Problemas conhecidos](#problemas-conhecidos)
- [Roadmap](#roadmap)
- [Próximos passos](#próximos-passos)
- [Repositórios relacionados](#repositórios-relacionados)
- [Histórico de sessões](#histórico-de-sessões)

---

## Arquitetura do pipeline

```
Isaac Sim (BRJGSILAGA, RTX 5090)
  ├─ FR5 + Robotiq 2F-85 (articulação única, 12 DOF)
  ├─ Câmera ZED X (asset óptico real, sem SDK C++ integrado)
  ├─ Cena de manipulação (5 parafusos M16, biblioteca IndustReal/Factory)
  └─ Painel de teleop (omni.ui) ──► grava PNG + estado de junta por ação
                                          │
                                          ▼
                    convert_to_lerobot.py (via `wsl -d weglabs` a partir do Isaac)
                                          │
                                          ▼
                         Dataset LeRobot (Parquet + MP4 AV1)
                                          │
                                          ▼
                    WSL2 (weglabs) ── lerobot-train --policy.path=smolvla_base
                                          │
                                          ▼
                         Checkpoint SmolVLA ── validação em loop fechado (sim)
                                          │
                                          ▼
                                  Deploy no FR5 real
```

Controle do Isaac Sim é feito via MCP server (repo irmão, ver
[Repositórios relacionados](#repositórios-relacionados)) — sem RMPflow/Lula;
o FR5 usa IK numérica implementada do zero (ver [Cinemática do FR5](#cinemática-do-fr5)).

## Status atual

*(última atualização: 2026-08-27, sessão 6 — ver [histórico completo](#histórico-de-sessões))*

**Concluído e funcionando:**

- **Câmera ZED X** montada com specs ópticas reais (Stereolabs, focal 2.2mm
  confirmado por cálculo). Posição final ajustada manualmente pelo usuário —
  fixa, não deve ser alterada. Path de captura:
  `.../ZED_X_model/base_link/ZED_X/CameraLeft`.
- **SDK completo da ZED compilado** (3 DLLs, incluindo Sim2Real), não
  integrado ainda — reservado para caso o gap sim→real de imagem se
  mostre relevante depois de testar no robô real.
- **Pipeline de gravação → dataset LeRobot validado ponta a ponta**: painel
  de teleop grava PNG + estado de junta por ação; `convert_to_lerobot.py`
  roda dentro do WSL2 chamado diretamente do Isaac Sim
  (`subprocess.run(["wsl", "-d", "weglabs", ...])`); dataset final
  recarregado e conferido.
- **Ensino por demonstração única** ("ensinar uma vez, funciona em qualquer
  parafuso"): o usuário posiciona o braço manualmente, marca a pose de
  referência, e o sistema recalcula a pose relativa para qualquer outro
  parafuso automaticamente. Confirmado generalizando corretamente em X/Z.

**Bloqueio aberto:**

- **Pega confiável da garra.** Três abordagens testadas (parafuso redondo em
  força alta/baixa, porca sextavada) — nenhuma segura de forma consistente.
  Causa é combinada (colisão de trajetória + velocidade/força de fechamento
  + alinhamento rotacional do objeto), não um parâmetro isolado. Decisão:
  não investir mais tempo agora — gerar o lote de treino aceitando sucesso
  parcial, validar o restante do pipeline (gravação → conversão → treino)
  em paralelo, e revisitar a taxa de pega depois.

**Pendência antes do próximo lote:** `Screw_3` foi removido da cena (virou o
teste da porca) e não foi recriado; `BOLT_PATHS` no painel ainda lista os 5
parafusos originais — `pick_all_and_record()` falha nessa iteração até isso
ser corrigido.

## Decisões técnicas

| Decisão | Racional |
|---|---|
| **SmolVLA como modelo principal** (não GR00T) | GR00T (3B) validado mas lento para iterar (~25–70s/step). SmolVLA (450M) roda ~100x mais rápido (3 steps/s) no mesmo hardware e mesmo formato de dataset — permite validar o pipeline em minutos. GR00T fica reservado para uma fase de produção/generalização. |
| **Fase intermediária no Franka pulada** | Como iterar com SmolVLA é barato, não compensa validar num robô intermediário antes de ir direto ao FR5. |
| **Coleta scripted/IK como método principal** | RMPflow/IK gera trajetórias em loop com pose do objeto randomizada — necessário para juntar 50+ episódios (teleop manual com poucas demos é causa documentada de falha em tutoriais comparáveis). |
| **Teleoperação como opção secundária** | Mantida para tarefas específicas que se beneficiem de demonstração manual direta, reaproveitando o padrão de teach-pendant já maduro no repo irmão. |
| **IK numérica própria, sem Lula/RMPflow** | Montar Lula do zero para um robô fora da biblioteca padrão do Isaac travou o projeto irmão por sessões inteiras com outro robô. IK via Jacobiano (mínimos quadrados amortecido) reaproveita uma receita já validada. |
| **Treino roda na mesma máquina do Isaac Sim (BRJGSILAGA)** | RTX 5090 com 32GB VRAM é suficiente para fine-tune leve (SmolVLA) mesmo com o Isaac Sim aberto (~23GB livres); treino pesado roda com o Isaac fechado. |
| **GR00T/LeRobot rodam em WSL2, não Windows nativo** | Rodar direto no Windows expôs ~7 bugs de plataforma (ver [Ambiente WSL2](#ambiente-wsl2)). Zero desses bugs em Linux nativo via WSL2 com passthrough de GPU. |

## Cinemática do FR5

O FR5 não faz parte da biblioteca padrão do Isaac Sim (208 robôs, sem
Fairino) — é carregado de um USD local. Cadeia cinemática (tipo DH) extraída
e validada contra a cena real:

| Joint | body0 → body1 | localPos0 (m) | localRot0 |
|---|---|---|---|
| j1 | base_link → shoulder_link | (0, 0, 0) | identidade |
| j2 | shoulder_link → upperarm_link | (0, 0, 0.152) | 90° X |
| j3 | upperarm_link → forearm_link | (-0.425, 0, 0) | identidade |
| j4 | forearm_link → wrist1_link | (-0.39501, 0, 0) | identidade |
| j5 | wrist1_link → wrist2_link | (0, 0, 0.1021) | 90° X |
| j6 | wrist2_link → wrist3_link | (0, 0, 0.102) | -90° X |
| TCP | wrist3_link → gripper (2F-85) | (0, 0, 0.09) | translação pura* |

\* A rotação -90° X documentada inicialmente para o mount wrist3→gripper
estava incorreta para fins de FK: os dois lados do `PhysicsFixedJoint` se
cancelam, resultando em translação pura. Verificado contra a cena ao vivo
(erro < 1e-4 m).

Estrutura equivalente a um UR5 (offsets de ombro/antebraço na mesma faixa),
sem surpresa cinemática. Implementação: FK por produto de matrizes
homogêneas (erro na 5ª casa decimal contra medição real); IK por
mínimos quadrados amortecido, `dq = Jᵀ(JJᵀ + λ²I)⁻¹·erro`, λ=0.05, erro de
orientação via *log map* de SO(3). Código em `demo/fr5_ik.py` no repo irmão.

**Limitações conhecidas do IK:**
- Ponto-a-ponto não evita colisão no caminho — saltos grandes no espaço de
  juntas podem cruzar zonas de colisão real mesmo com origem e destino
  válidos.
- `j1` tem zona morta mecânica (±175°, não gira 360°) — normalizar cada
  junta para o valor mais próximo *dentro do limite*, não apenas o mais
  próximo matematicamente.
- Colisão física repetível quando `j2` gira para a faixa -150° a -200°
  (forearm passa perto do chão) — sem checagem de colisão real implementada
  ainda; mitigado até agora reposicionando a cena para reduzir o
  deslocamento angular necessário.

## Infraestrutura

- **Isaac Sim 6.0.1**, remoto no host `BRJGSILAGA`, controlado via MCP
  server ([repo irmão](#repositórios-relacionados)).
- **GPU**: RTX 5090, 32GB VRAM (medido via `nvidia-smi`; ~9,5GB em uso pelo
  Isaac Sim quando aberto).
- **Fine-tuning** roda na mesma máquina. Com o Isaac Sim fechado, os 32GB
  ficam livres; com o Isaac aberto, usar `--no-tune_diffusion_model` para
  caber nos ~23GB restantes.

### Ambiente WSL2

GR00T é suportado apenas em Linux/Jetson — rodar direto no Windows expôs uma
cadeia de bugs de plataforma (paths com `\` quebrando repo-id do HuggingFace,
PyTorch CPU-only por padrão, `multiprocessing_context="fork"` hardcoded,
incompatibilidade FFmpeg 8/torchcodec, conflito de PATH entre instalações de
FFmpeg, build incorreto do FFmpeg). Migrado para **WSL2 Ubuntu 22.04**
(distro `weglabs`) — GPU com passthrough nativo via driver Windows, zero
desses bugs.

- Reset de senha sem credencial atual: `wsl -u root -d weglabs` → `passwd <user>`.
- GR00T: clonado em `~/Isaac-GR00T`; `deepspeed` removido do `pyproject.toml`
  (exige CUDA Toolkit completo, desnecessário para treino single-GPU).
  Requer `git lfs install && git lfs pull` após o clone (wheels locais vêm
  como ponteiro LFS).
- LeRobot/SmolVLA: clonado em `~/lerobot`, venv própria (`uv venv`),
  instalado com `pip install -e ".[smolvla,dataset]"`.
- Treino: `lerobot-train --policy.path=lerobot/smolvla_base
  --dataset.repo_id=<repo>`. Se o dataset tiver menos câmeras que o
  pré-treino (3 esperadas): `--policy.empty_cameras=N` +
  `--rename_map='{"chave_origem": "observation.images.camera1"}'`.

## Problemas conhecidos

- **Pega da garra não confiável** (bloqueio atual, ver [Status atual](#status-atual)).
- **Pose do robô não persiste entre sessões do Isaac** — `set_joint_positions`
  não é salvo no `.usd`. Sempre reler `get_joint_positions` no início de
  cada sessão em vez de assumir a última pose registrada.
- **Estado de junta corrompido pode ser salvo no `.usd`** — já ocorreu de um
  travamento sobreviver a reabrir o stage e a reiniciar o Isaac Sim
  inteiro. Único fix confirmado: deletar o robô e recarregar o `fr5.usd`
  original do zero.
- **Nunca misturar controle via atributo USD bruto
  (`drive:angular:physics:targetPosition`) com `set_joint_positions` do
  MCP** — os dois caminhos dessincronizam entre si (mesmo depois de
  Stop→Play, um deles para de responder). Usar exclusivamente as MCP
  tools para controle de junta.
- **Mudanças estruturais de física** (joint, `ArticulationRootAPI`) às vezes
  exigem **dois** ciclos Stop→Play para o solver reparsear de verdade —
  sempre confirmar com `step_simulation`/`observe_prims` antes de seguir.
- **Topo de colisão da mesa ≠ topo visual**: mesa tem caixotes decorativos
  sem física que inflam o bounding box do grupo. Colisão real via
  `convexDecomposition` no submesh correto fica em `z = 0.3750`, não
  `0.4007`.
- **Assets com `scale` + `translate`**: a composição é multiplicativa, não
  aditiva — `translate = posição_mundial_desejada / escala`.
- **Extensão nativa do SDK ZED (streaming C++) não instalada**: o build
  exige internet no host Isaac (bloqueado) e o ZED SDK 5.4.1+ instalado
  localmente. Usa-se apenas o asset óptico USD (FOV/geometria corretos);
  perde-se apenas o pós-processamento "Sim2Real" (vinheta, ruído,
  auto-exposição) — revisitar se o gap sim→real de imagem se mostrar
  relevante no robô real.

## Roadmap

- [x] **Fase A — Pipeline de treino validado**: fine-tune ponta a ponta
      rodado tanto para GR00T (dataset `cube_to_bowl_5`) quanto SmolVLA
      (`lerobot/pusht`, 1000 steps, 3,14 steps/s, 10GB VRAM).
- [ ] **Fase B — Fairino FR5 no Isaac Sim** *(atual)*
  - [x] Montagem do gripper Robotiq 2F-85 no flange (fixed joint,
        articulação única, 12 DOF)
  - [x] Cena de manipulação (parafusos M16, física validada)
  - [x] IK numérica (FK + Jacobiano amortecido) implementada e validada
  - [x] Câmera ZED X (asset óptico) integrada
  - [x] Pipeline de gravação → conversão LeRobot validado ponta a ponta
  - [x] Ensino por demonstração única (posição relativa ao objeto)
  - [ ] Pega confiável da garra
  - [ ] Geração do lote de treino (50+ episódios)
  - [ ] Fine-tune SmolVLA no embodiment FR5
  - [ ] Validação em loop fechado no Isaac Sim
- [ ] **Fase C — Deploy no robô real**

## Próximos passos

1. Recriar `Screw_3` ou remover de `BOLT_PATHS` antes de rodar
   `pick_all_and_record()`.
2. Rodar o lote de gravação nos parafusos como estão (aceitando sucesso
   parcial de pega) para validar o pipeline completo gravação → conversão
   → `lerobot-train`.
3. Melhorar a taxa de pega da garra em paralelo (candidatos: checagem de
   colisão real nos waypoints de descida, ajuste de força/velocidade de
   fechamento, alinhamento rotacional do objeto antes de fechar).
4. Fine-tune SmolVLA no dataset gerado e validar em loop fechado no Isaac
   Sim antes de portar para o robô real.

## Repositórios relacionados

- `isaacsim-mcp-server-remote` (repo irmão local, não publicado neste link) —
  MCP server que controla o Isaac Sim remoto; contém o código reaproveitável
  (`demo/fr5_ik.py`, painel de teleop, pick-and-place do Franka verificado)
  e as notas técnicas detalhadas (`agent-notes/`) referenciadas neste README.
- [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [HuggingFace LeRobot / SmolVLA](https://github.com/huggingface/lerobot)
- [Stereolabs ZED Isaac Sim](https://github.com/stereolabs/zed-isaac-sim)

## Histórico de sessões

Registro cronológico detalhado de cada sessão de trabalho — decisões,
armadilhas encontradas e o estado exato em que cada uma terminou. Preservado
como referência técnica; para o estado atual, ver [Status atual](#status-atual).

<details>
<summary><strong>Sessão 6 (2026-08-27) — Câmera ZED X, pipeline LeRobot ponta a ponta</strong></summary>

Câmera ZED X real montada (specs Stereolabs, posição ajustada manualmente
pelo usuário — fixa). Pipeline gravação→LeRobot testado ponta a ponta:
painel de teleop grava PNG+junta, `convert_to_lerobot.py` roda no WSL2
(achado técnico: `wsl -d weglabs` pode ser chamado direto de dentro do
Isaac via `subprocess`, sem ferramenta de shell separada) e produz dataset
LeRobot real (Parquet+MP4). Sistema de "ensinar uma vez" implementado e
confirmado funcionando para posicionamento — mas pega confiável da garra
continua sem resolver depois de 3 tentativas reais (força, velocidade,
formato do objeto). Decisão: parar de perseguir pega perfeita, gerar lote
com sucesso parcial e validar o pipeline de treino primeiro. Detalhe
completo em `agent-notes/fr5-teach-and-grasp-reliability.md` e
`agent-notes/fr5-lerobot-recording.md` no repo irmão.

</details>

<details>
<summary><strong>Sessão 5 (2026-08-27) — Troca de tarefa: caixa → parafusos M16</strong></summary>

Isaac reconectado limpo (`BRJGSILAGA:8767`). Cena confirmada pronta (robô
12 DOF, base ok). Tarefa mudou de "pick de caixa" para "manipular
parafusos": caixa deletada, substituída por 5 parafusos M16 reais da
biblioteca IndustReal/Factory (melhor que YCB para precisão). Achado
importante: o topo *visual* da mesa (0.4007) difere do topo de *colisão*
real (0.3750) — a mesa tem caixotes decorativos sem física que inflam o
bbox do grupo. Corrigido trocando `boundingCube`→`convexDecomposition` no
submesh certo. 5 parafusos posicionados e verificados fisicamente
estáveis. Parafusos vêm soldados no mundo por padrão (`root_joint` fixo) —
precisam ser desativados para virar objeto pegável. Um estado de junta
corrompido (`j4` lendo 252°, fora do limite de 85°) ficou gravado no
próprio arquivo `.usd`, sobrevivendo a reabrir o stage e reiniciar o Isaac
Sim inteiro — único fix foi deletar o robô e recarregar `fr5.usd` original.
Receita completa em `agent-notes/fr5-bolt-manipulation.md` no repo irmão.

</details>

<details>
<summary><strong>Sessão 4 (2026-08-27) — Troca para teleoperação, montagem do gripper</strong></summary>

Usuário pediu troca para teleoperação nesta demonstração de pick (IK
scripted ficava bom em posição mas ruim em orientação). Painel
`fr5_teleop_panel_v1.py` construído. Mesa/caixa recriadas com assets reais
(`packing_table.usd` + YCB `cracker_box`) — achado importante: nesses
assets, `translate` compõe multiplicativamente com `scale`, não
aditivamente. Achado grave: um estado de junta corrompido (posição real
travada fora do próprio limite físico) ficou gravado no `.usd`,
sobrevivendo a reabrir o stage e reiniciar o app inteiro — único fix foi
deletar o robô e recarregar `fr5.usd` limpo. Remontagem da garra 2F-85
(receita corrigida: translação pura, sem rotação) coincidiu com um
terceiro travamento do Isaac na mesma sessão.

</details>

<details>
<summary><strong>Sessão 3 (2026-08-26) — IK resolvido, primeiro grasp (sem segurar)</strong></summary>

Colisão nova (sessão 2) identificada como auto-colisão real (antebraço vs
garra), corrigida com IK só-de-posição e limite `j2 >= -50°`. Problema
maior encontrado: `j1` trava girando além de ~-120° para um lado só, mas é
livre no outro — não é auto-colisão nem mesa. Causa raiz não identificada,
mas um branch de IK alternativo (cotovelo invertido, `j1≈+55°`) funcionou:
zero erro em 9 passos, braço chegou limpo sobre a caixa e desceu sem
travar. Fechou a garra, fez contato, mas empurrou a caixa em vez de segurá-la
(não subiu junto ao levantar) — geometria de grasp ainda não centralizada
entre os dedos. Problema de movimento/colisão resolvido; faltava só a
geometria do grasp. Receita completa em `agent-notes/fr5-pick-fk-ik.md`
(repo irmão), seção "Session 3".

</details>

<details>
<summary><strong>Sessão 2 (2026-08-26) — FK/IK implementados e validados</strong></summary>

Código FK/IK da sessão anterior nunca tinha sido salvo de verdade
(`execute_script` não persiste nada) — reconstruído e salvo em
`demo/fr5_ik.py` no repo irmão. Bug corrigido: mount wrist3→gripper é
translação pura (0,0,0.09), sem a rotação -90°X documentada
originalmente (os dois lados do fixed joint se cancelam). Achado
operacional: pose do robô não sobrevive entre sessões
(`set_joint_positions` não salva no `.usd`) — sempre reler
`get_joint_positions` no início. Checagem de colisão via `position_error`
estável confirmada como método funcional. FK validado contra física real
(erro na 5ª casa decimal); IK converge bem para deslocamentos pequenos.
Achado: IK ponto-a-ponto não evita colisão no caminho — waypoints seriam
necessários (não implementado nesta sessão). Dois `ActionGraph` de ponte
ROS2 embutidos na cena original foram desativados (`SetActive(False)`,
reversível) por interferir no controle direto.

</details>

<details>
<summary><strong>Waypoints e colisão (2026-08-26) — detalhe técnico</strong></summary>

Interpolação em linha reta (cartesiana e depois espaço de juntas) achou
dois problemas reais:

1. **Zona morta mecânica do `j1`** — não gira os 360° completos, só ±175°.
   Um caminho reto pode exigir cruzar exatamente esse ~10° inexistente.
   Fix: normalizar cada junta para o valor **dentro do limite** mais
   próximo do anterior (testar ±360°), não apenas "mais próximo" sem
   checar limite.
2. **Colisão física real em giros grandes de `j2`** — em pelo menos dois
   testes independentes, `j2` travou com erro de posição grande e estável
   entre -150° e -200°, sempre na mesma faixa. Suspeita: `forearm_link`
   passa muito perto do chão (z~0.11m) nessa faixa. Não resolvido — requer
   checagem de colisão real (não só limite de junta) antes de aceitar um
   waypoint.

**Armadilha grave**: tentar acelerar o loop de waypoints escrevendo o
atributo USD `drive:angular:physics:targetPosition` direto + chamando
`omni.physx.get_physx_simulation_interface().simulate()` manualmente
dessincroniza do caminho que `set_joint_positions` do MCP usa por baixo —
depois disso, `set_joint_positions` parou de afetar `j1`/`j2` mesmo após
Stop+Play. Nunca misturar os dois caminhos de controle na mesma
sessão/cena. Recuperado escrevendo o atributo bruto direto para um valor
seguro + Stop/Play.

**Fix mais simples que evitar colisão**: a causa raiz de boa parte disso
era a mesa estar posicionada no quadrante oposto ao alcance natural do
robô a partir da pose seed (`-Y` em vez de `+Y`), forçando giros grandes
(>100°) desnecessários. Reposicionar a mesa+caixa para o mesmo lado que o
robô já alcança confortavelmente reduziu o deslocamento de junta necessário
de ~150° para ~36°, convergindo sem colisão. Lição: antes de implementar um
planejador de colisão completo, vale checar se o layout da cena está
simplesmente mal posicionado para o alcance natural do robô. A descida
final (~7cm de folga entre gripper e topo do objeto) seguiu travando mesmo
depois desse fix — confirma que o problema é proximidade de colisão real
perto do alvo, não o tamanho do passo.

</details>

<details>
<summary><strong>Montagem do gripper Robotiq 2F-85 (2026-08-26) — receita reaproveitável</strong></summary>

1. Obter a transform mundial completa do link do flange (posição +
   quatérnio) via
   `UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)` —
   `get_prim_info` só retorna posição, não rotação.
2. `load_usd` do gripper numa prim separada; setar a transform dele
   (translate + orient) para igualar exatamente a do flange.
3. Criar um `UsdPhysics.FixedJoint` com `body0` = link do flange,
   `body1` = link base do gripper, local pos/rot identidade dos dois lados
   (já coincidentes em mundo).
4. **Armadilha**: o asset do gripper vem com sua própria
   `ArticulationRootAPI` — dois roots ligados por joint fixo quebra
   `get_robot_info` (`'NoneType' object has no attribute 'is_homogeneous'`).
   Fix: `prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)` na raiz do
   gripper, seguido de **Stop + Play** (mudança estrutural não aplica em
   runtime). Resultado: articulação única, 12 DOF (6 braço + 6 gripper —
   `finger_joint` é o drive principal do 2F-85).
5. A transform mundial do flange não é a orientação correta para montar o
   gripper diretamente — o eixo local do `wrist3_link` não bate com a face
   física do flange. Parâmetros finais, verificados visual e fisicamente,
   no joint `.../wrist3_link/gripper_fixed_joint`:
   - `physics:localRot0 = (0.7071, -0.7071, 0, 0)` (-90° X)
   - `physics:localPos0 = (0, 0, 0.09)` (9cm ao longo do Z local do
     flange, após a rotação acima)
   - `physics:localPos1` / `physics:localRot1` = identidade
   - Específico deste FR5 + este 2F-85 — recalcular se o robô ou o
     gripper mudar.

**Armadilhas de física em runtime:**
- Editar transform de um corpo rígido com a simulação tocando faz a física
  sobrescrever a edição no frame seguinte com um valor corrompido. Sempre
  **Stop** antes de editar transform ou atributo de joint.
- Editar `localPos0`/`localRot0` às vezes exige **dois** ciclos
  Stop→Play→Stop→Play para o solver reparsear — sempre confirmar via
  `observe_prims` antes de assumir que a edição colou.
- O erro `'NoneType' object has no attribute 'is_homogeneous'` no stderr
  durante edições de física é ruído de UI, não indica falha real — só
  importa se `get_robot_info` continuar quebrando depois do Play.

</details>

