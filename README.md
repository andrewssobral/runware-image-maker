Image Maker
---
### Welcome to the beginnings of the best image generation service in the world!

The current directory layout is as follows:
 - `image_maker/` | The server source code
 - `lib_image_maker/` | The upstream image generation model code.
   - This is a fork which will be updated independently of the server.
 - `main.py` | The server entrypoint
 - `monitor.py` | Small script to watch VRAM
 - `test.py` | The `pytest` suite

TODOs
---
### Flux.1 just launched, and we need it integrated ASAP.

Specifically, there's two new models:
```
"black-forest-labs/FLUX.1-dev"
"black-forest-labs/FLUX.1-schnell"
```
both of which should work on our current GPUs.

If this is your first time working in this repository, start with `image_maker/schema.py` to get a feel for the current model capabilities.

### Performance is king!
We're always looking for ways to improve performance, reduce overhead, and ensure optimal GPU utilization.

CONTRIBUTING
---
- **Submission Format:**
  - If you don't have repo access: submit patch files + a self-review summarizing your changes
  - **For first time contributors**, we'd appreciate taking the time to review the system and including a design doc.
    - If you could redesign this system, what would you change? API design, memory management, testing strategy...
- **AI Usage:** Must be disclosed. Ideally we would like to see:
  - Prompts used for any automated tasks or substantial code generation.
  - Models and harnesses / tools used.
  - An overview of your strategy in using LLMs or other coding agents.
- **Questions:** If something is unclear, document your assumptions rather than asking for clarification.
- **Testing:** Existing tests should pass. Add tests if you think they're needed and explain your reasoning in the self-review.
