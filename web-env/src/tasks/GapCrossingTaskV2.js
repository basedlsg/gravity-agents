/**
 * GapCrossingTask V2 - Calibrated for meaningful difficulty
 *
 * Changes from V1:
 * - Wider gap (4.5m vs 3.0m) - requires precise timing
 * - Lower jump (1.2m vs 2.0m) - less margin for error
 * - Reduced air control - commits to trajectory
 * - Run-up bonus - rewards understanding of momentum
 * - Narrower platforms - punishes lateral drift
 *
 * Target success rates:
 * - NRL-F at 1g: 25-40%
 * - RL agents at 1g after training: 70-85%
 */

import { BaseTask } from './BaseTask.js';

export class GapCrossingTaskV2 extends BaseTask {
  constructor(world, config) {
    super(world, config);

    // V2 CALIBRATED PARAMETERS
    // Gap configuration - HARDER (requires good timing)
    // Max jump distance is ~5.15m with perfect air control
    // Setting gap to 4.6m gives ~0.5m margin when optimal
    this.gapWidth = config.gapWidth ?? 4.6;        // Requires good timing
    this.gapVariance = config.gapVariance ?? 0.1;  // Minimal variance for consistency

    // Platform configuration - NARROWER
    this.platformWidth = 4.0;   // Was 5.0 - less runway
    this.platformDepth = 3.0;   // Was 5.0 - narrower
    this.platformHeight = 0.5;

    // V2.2: Configurable goal platform length (for 0.5g overshoot accommodation)
    // At 0.5g, jump@6 lands at ~17m, so we need platform to extend to ~20m
    this.goalPlatformWidth = config.goalPlatformWidth ?? this.platformWidth;

    // V2.1: Configurable landing zone for Task A vs Task B experiments
    // Task A (invariant): wide landing zone (default)
    // Task B (adaptive): narrow landing zone
    this.landingZoneWidth = config.landingZoneWidth ?? null;  // null = use platform width
    this.landingZoneStart = config.landingZoneStart ?? null;  // Custom start position
    this.landingZoneEnd = config.landingZoneEnd ?? null;      // Custom end position

    // Agent parameters - NERFED
    this.agentRadius = 0.4;
    this.agentHeight = 1.8;
    this.agentMass = 80;        // Was 70
    this.moveSpeed = 3.0;       // Was 4.0 - slower
    this.maxSpeed = 4.0;        // Was 5.0

    // Jump physics - CRITICAL CHANGES
    this.jumpHeight = 1.2;      // Was 2.0 - lower jump
    this.jumpCooldownMs = 500;  // Prevent bunny hopping
    this.airControl = 0.3;      // Was 1.0 - much less air control
    this.runUpBonus = 1.3;      // 30% bonus with momentum

    // Movement state
    this.isGrounded = false;
    this.canJump = true;
    this.lastJumpTime = 0;
    this.currentAction = 'idle';  // Track current action for continuous application

    // Action mapping
    this.actions = ['forward', 'back', 'left', 'right', 'jump', 'idle'];

    this.setup();
  }

  setup() {
    // Calculate actual gap with variance
    const actualGap = this.gapWidth + this.randomInRange(-this.gapVariance, this.gapVariance);

    // Platform A (start) - centered at origin
    this.world.createBox('platformA', {
      position: [0, -this.platformHeight / 2, 0],
      size: [this.platformWidth, this.platformHeight, this.platformDepth],
      mass: 0
    });

    // Platform B (goal) - across the gap
    // V2.2: Use goalPlatformWidth for the goal platform (can be longer for invariant task)
    const platformBX = this.platformWidth / 2 + actualGap + this.goalPlatformWidth / 2;
    this.world.createBox('platformB', {
      position: [platformBX, -this.platformHeight / 2, 0],
      size: [this.goalPlatformWidth, this.platformHeight, this.platformDepth],
      mass: 0
    });

    // Goal zone - configurable for Task A (wide) vs Task B (narrow)
    let goalMinX, goalMaxX;
    if (this.landingZoneStart !== null && this.landingZoneEnd !== null) {
      // Custom landing zone specified
      goalMinX = this.landingZoneStart;
      goalMaxX = this.landingZoneEnd;
    } else if (this.landingZoneWidth !== null) {
      // Width specified, center it on platform
      const halfWidth = this.landingZoneWidth / 2;
      goalMinX = platformBX - halfWidth;
      goalMaxX = platformBX + halfWidth;
    } else {
      // Default: most of the platform
      goalMinX = platformBX - this.platformWidth / 2 + 0.8;
      goalMaxX = platformBX + this.platformWidth / 2 - 0.8;
    }

    this.goalZone = {
      minX: goalMinX,
      maxX: goalMaxX,
      minZ: -this.platformDepth / 2 + 0.8,
      maxZ: this.platformDepth / 2 - 0.8,
      minY: 0,
      maxY: 2.0
    };

    // Store gap info for observations
    this.gapStart = this.platformWidth / 2;
    this.gapEnd = this.gapStart + actualGap;
    this.actualGapWidth = actualGap;

    // Agent (capsule-like body) - starts further back for run-up
    const agentStartX = -this.platformWidth / 3;
    const agentBody = this.world.createCapsule('agent', {
      position: [agentStartX, this.agentHeight / 2 + 0.1, 0],
      radius: this.agentRadius,
      height: this.agentHeight,
      mass: this.agentMass
    });

    // Remove friction from agent so velocity persists
    agentBody.linearDamping = 0;
  }

  getObservation() {
    const agentState = this.world.getBodyState('agent');

    return {
      // Agent state
      agentPosition: agentState.position,
      agentVelocity: agentState.velocity,

      // Task info
      gapStart: this.gapStart,
      gapEnd: this.gapEnd,
      gapWidth: this.actualGapWidth,
      goalZone: this.goalZone,

      // Physics info
      gravity: this.config.gravity,
      isGrounded: this.isGrounded,

      // V2: Additional physics info for agents
      jumpHeight: this.jumpHeight,
      moveSpeed: this.moveSpeed,
      airControl: this.airControl,
      runUpBonus: this.runUpBonus,

      // Available actions
      actions: this.actions
    };
  }

  applyAction(action) {
    const agent = this.world.getBody('agent');
    if (!agent) return;

    const now = Date.now();

    // Check if grounded
    const wasGrounded = this.isGrounded;
    this.isGrounded = Math.abs(agent.velocity.y) < 0.1 && agent.position.y < this.agentHeight / 2 + 0.3;

    // Reset jump cooldown when landing
    if (!wasGrounded && this.isGrounded) {
      this.canJump = true;
    }

    // Normalize action
    const actionStr = typeof action === 'number' ? this.actions[action] : action.toLowerCase();

    // Apply action
    switch (actionStr) {
      case 'forward':
        if (this.isGrounded) {
          // Full control on ground - set velocity directly
          agent.velocity.x = this.moveSpeed;
        } else {
          // Reduced air control
          agent.velocity.x += this.moveSpeed * this.airControl * 0.1;
          agent.velocity.x = Math.min(agent.velocity.x, this.maxSpeed);
        }
        break;

      case 'back':
        if (this.isGrounded) {
          agent.velocity.x = -this.moveSpeed;
        } else {
          agent.velocity.x -= this.moveSpeed * this.airControl * 0.1;
          agent.velocity.x = Math.max(agent.velocity.x, -this.maxSpeed);
        }
        break;

      case 'left':
        if (this.isGrounded) {
          agent.velocity.z = -this.moveSpeed * 0.8;
        } else {
          agent.velocity.z -= this.moveSpeed * this.airControl * 0.05;
        }
        break;

      case 'right':
        if (this.isGrounded) {
          agent.velocity.z = this.moveSpeed * 0.8;
        } else {
          agent.velocity.z += this.moveSpeed * this.airControl * 0.05;
        }
        break;

      case 'jump':
        if (this.isGrounded && this.canJump && (now - this.lastJumpTime) > this.jumpCooldownMs) {
          // Base jump velocity for desired height
          // v = sqrt(2 * g * h)
          let jumpVelocity = Math.sqrt(2 * this.config.gravity * this.jumpHeight);

          // V2: Run-up bonus - jumping while moving forward gives bonus
          const forwardSpeed = agent.velocity.x;
          if (forwardSpeed > this.moveSpeed * 0.5) {
            jumpVelocity *= this.runUpBonus;
          }

          agent.velocity.y = jumpVelocity;
          this.canJump = false;
          this.lastJumpTime = now;
        }
        break;

      case 'idle':
      default:
        // Apply drag
        if (this.isGrounded) {
          agent.velocity.x *= 0.85;
          agent.velocity.z *= 0.85;
        } else {
          // Less drag in air
          agent.velocity.x *= 0.98;
          agent.velocity.z *= 0.98;
        }
        break;
    }

    // Clamp horizontal velocity
    agent.velocity.x = Math.max(-this.maxSpeed, Math.min(this.maxSpeed, agent.velocity.x));
    agent.velocity.z = Math.max(-this.maxSpeed, Math.min(this.maxSpeed, agent.velocity.z));
  }

  checkTermination() {
    const agent = this.world.getBody('agent');
    if (!agent) {
      return { done: true, success: false, reason: 'agent_missing' };
    }

    const pos = agent.position;

    // Check if fell into gap or off platforms
    if (pos.y < -5) {
      return { done: true, success: false, reason: 'fell' };
    }

    // V2: Check if fell off sides (narrower platform)
    if (Math.abs(pos.z) > this.platformDepth / 2 + 0.5) {
      return { done: true, success: false, reason: 'fell_side' };
    }

    // Check if in goal zone
    const inGoal =
      pos.x >= this.goalZone.minX && pos.x <= this.goalZone.maxX &&
      pos.z >= this.goalZone.minZ && pos.z <= this.goalZone.maxZ &&
      pos.y >= this.goalZone.minY && pos.y <= this.goalZone.maxY;

    if (inGoal) {
      this.successSteps++;
      if (this.successSteps >= this.requiredSuccessSteps) {
        return { done: true, success: true, reason: 'reached_goal' };
      }
    } else {
      this.successSteps = 0;
    }

    return { done: false, success: false, reason: 'ongoing' };
  }

  getReward(done, success) {
    if (success) {
      return 1.0;
    }
    return 0.0;
  }

  // V2: Physics calculation helper for prompts
  getPhysicsInfo() {
    const g = this.config.gravity;
    const jumpVel = Math.sqrt(2 * g * this.jumpHeight);
    const jumpVelWithRunUp = jumpVel * this.runUpBonus;
    const timeInAir = 2 * jumpVel / g;
    const timeInAirWithRunUp = 2 * jumpVelWithRunUp / g;

    return {
      gravity: g,
      jumpVelocity: jumpVel,
      jumpVelocityWithRunUp: jumpVelWithRunUp,
      timeInAir: timeInAir,
      timeInAirWithRunUp: timeInAirWithRunUp,
      horizontalDistNoRunUp: this.moveSpeed * timeInAir,
      horizontalDistWithRunUp: this.moveSpeed * timeInAirWithRunUp,
      gapWidth: this.actualGapWidth
    };
  }
}
