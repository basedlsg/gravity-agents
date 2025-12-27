/**
 * Gravity Agents - Physics Environment Server
 *
 * Exposes reset/step API for RL agents to interact with
 * three.js + cannon-es physics simulation.
 */

import express from 'express';
import cors from 'cors';
import { WebSocketServer } from 'ws';
import { createServer } from 'http';
import { PhysicsWorld } from './src/physics/PhysicsWorld.js';
import { GapCrossingTask } from './src/tasks/GapCrossingTask.js';
import { ThrowBlockTask } from './src/tasks/ThrowBlockTask.js';
import { GapCrossingTaskV2 } from './src/tasks/GapCrossingTaskV2.js';
import { ThrowBlockTaskV2 } from './src/tasks/ThrowBlockTaskV2.js';

const app = express();
app.use(cors());
app.use(express.json());

const server = createServer(app);
const wss = new WebSocketServer({ server });

// Active environments per session
const environments = new Map();

// Task registry - includes V1 and V2 versions
const TASKS = {
  gap: GapCrossingTask,
  throw: ThrowBlockTask,
  // V2 calibrated tasks
  'gap_v2': GapCrossingTaskV2,
  'throw_v2': ThrowBlockTaskV2
};

// Default config
const DEFAULT_CONFIG = {
  task: 'gap',
  gravity: 9.81,
  seed: 42,
  maxSteps: 500,
  physicsTicksPerStep: 10,
  timeStep: 1 / 60
};

/**
 * Create or reset an environment
 */
function createEnvironment(sessionId, config) {
  const mergedConfig = { ...DEFAULT_CONFIG, ...config };

  // Create physics world
  const world = new PhysicsWorld({
    gravity: mergedConfig.gravity,
    timeStep: mergedConfig.timeStep
  });

  // Determine task name (with optional V2 suffix)
  let taskName = mergedConfig.task;
  if (mergedConfig.taskVersion === 'v2') {
    taskName = `${mergedConfig.task}_v2`;
  }

  // Create task
  const TaskClass = TASKS[taskName];
  if (!TaskClass) {
    throw new Error(`Unknown task: ${taskName} (available: ${Object.keys(TASKS).join(', ')})`);
  }

  const task = new TaskClass(world, mergedConfig);

  const env = {
    world,
    task,
    config: mergedConfig,
    stepCount: 0,
    done: false,
    totalReward: 0
  };

  environments.set(sessionId, env);

  return task.getObservation();
}

/**
 * Step the environment
 */
function stepEnvironment(sessionId, action, params = {}) {
  // console.log(`StepEnv params:`, params);
  const env = environments.get(sessionId);
  if (!env) {
    throw new Error(`No environment for session: ${sessionId}`);
  }

  if (env.done) {
    return {
      observation: env.task.getObservation(),
      reward: 0,
      done: true,
      info: { reason: 'already_done' }
    };
  }

  // Step physics - apply action each tick to maintain velocity against friction
  for (let i = 0; i < env.config.physicsTicksPerStep; i++) {
    env.task.applyAction(action, params);
    env.world.step();
  }

  env.stepCount++;

  // Check termination
  const { done, success, reason } = env.task.checkTermination();
  const reward = env.task.getReward(done, success);

  env.done = done || env.stepCount >= env.config.maxSteps;
  env.totalReward += reward;

  const observation = env.task.getObservation();

  return {
    observation,
    reward,
    done: env.done,
    info: {
      step: env.stepCount,
      success,
      reason: env.done && !done ? 'timeout' : reason,
      totalReward: env.totalReward
    }
  };
}

// REST API endpoints

app.post('/reset', (req, res) => {
  try {
    const sessionId = req.body.sessionId || 'default';
    // Accept config either as nested object or flat in body
    const config = req.body.config || {
      task: req.body.task,
      taskVersion: req.body.taskVersion,
      gravity: req.body.gravity,
      seed: req.body.seed,
      // V2.1+: Landing zone configuration
      landingZoneStart: req.body.landingZoneStart,
      landingZoneEnd: req.body.landingZoneEnd,
      landingZoneWidth: req.body.landingZoneWidth,
      // V2.2: Goal platform width (for extended platforms)
      goalPlatformWidth: req.body.goalPlatformWidth
    };
    const observation = createEnvironment(sessionId, config);
    res.json({ success: true, observation });
  } catch (error) {
    res.status(400).json({ success: false, error: error.message });
  }
});

app.post('/step', (req, res) => {
  try {
    const sessionId = req.body.sessionId || 'default';
    const action = req.body.action;
    if (action === undefined) {
      throw new Error('Action is required');
    }
    const params = req.body; // Pass all body params (includes durationScale)
    const result = stepEnvironment(sessionId, action, params);
    res.json({ success: true, ...result });
  } catch (error) {
    res.status(400).json({ success: false, error: error.message });
  }
});

app.get('/info', (req, res) => {
  res.json({
    tasks: Object.keys(TASKS),
    actions: {
      gap: ['forward', 'back', 'left', 'right', 'jump', 'idle'],
      throw: ['forward', 'back', 'left', 'right', 'pick', 'drop', 'throw_weak', 'throw_medium', 'throw_strong', 'idle']
    },
    defaultConfig: DEFAULT_CONFIG
  });
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok', environments: environments.size });
});

// WebSocket for real-time streaming (optional viewer)
wss.on('connection', (ws) => {
  console.log('Viewer connected');

  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message);
      if (data.type === 'subscribe') {
        ws.sessionId = data.sessionId || 'default';
      }
    } catch (e) {
      console.error('WebSocket message error:', e);
    }
  });

  ws.on('close', () => {
    console.log('Viewer disconnected');
  });
});

// Broadcast state updates to viewers
function broadcastState(sessionId, state) {
  wss.clients.forEach((client) => {
    if (client.sessionId === sessionId && client.readyState === 1) {
      client.send(JSON.stringify(state));
    }
  });
}

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Gravity Agents Environment Server running on port ${PORT}`);
  console.log(`REST API: http://localhost:${PORT}`);
  console.log(`WebSocket: ws://localhost:${PORT}`);
});

export { createEnvironment, stepEnvironment };
