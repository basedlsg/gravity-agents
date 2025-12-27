/**
 * GapCrossingTask - Agent must jump across a gap between two platforms
 *
 * Actions: forward, back, left, right, jump, idle
 * Success: Agent's body center inside goal area for N consecutive steps
 */

import { BaseTask } from './BaseTask.js';

export class GapCrossingTask extends BaseTask {
  constructor(world, config) {
    super(world, config);

    // Task-specific config
    this.gapWidth = config.gapWidth ?? 3.0; // Base gap width
    this.gapVariance = config.gapVariance ?? 0.5; // Random variance
    this.platformWidth = 5.0;
    this.platformDepth = 5.0;
    this.platformHeight = 0.5;

    // Agent parameters
    this.agentRadius = 0.4;
    this.agentHeight = 1.8;
    this.agentMass = 70;
    this.moveForce = 500;
    this.jumpImpulse = 350;

    // Movement state
    this.isGrounded = false;
    this.canJump = true;

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
    const platformBX = this.platformWidth / 2 + actualGap + this.platformWidth / 2;
    this.world.createBox('platformB', {
      position: [platformBX, -this.platformHeight / 2, 0],
      size: [this.platformWidth, this.platformHeight, this.platformDepth],
      mass: 0
    });

    // Goal zone (invisible, for detection)
    this.goalZone = {
      minX: platformBX - this.platformWidth / 2 + 0.5,
      maxX: platformBX + this.platformWidth / 2 - 0.5,
      minZ: -this.platformDepth / 2 + 0.5,
      maxZ: this.platformDepth / 2 - 0.5,
      minY: 0,
      maxY: 2.0
    };

    // Store gap info for observations
    this.gapStart = this.platformWidth / 2;
    this.gapEnd = this.gapStart + actualGap;
    this.actualGapWidth = actualGap;

    // Agent (capsule-like body)
    const agentStartX = -this.platformWidth / 4;
    this.world.createCapsule('agent', {
      position: [agentStartX, this.agentHeight / 2 + 0.1, 0],
      radius: this.agentRadius,
      height: this.agentHeight,
      mass: this.agentMass
    });
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

      // Available actions
      actions: this.actions
    };
  }

  applyAction(action) {
    const agent = this.world.getBody('agent');
    if (!agent) return;

    // Check if grounded (simple Y velocity check + position check)
    this.isGrounded = Math.abs(agent.velocity.y) < 0.1 && agent.position.y < this.agentHeight / 2 + 0.3;

    // Normalize action
    const actionStr = typeof action === 'number' ? this.actions[action] : action.toLowerCase();

    // Apply action
    switch (actionStr) {
      case 'forward':
        agent.velocity.x = 4.0;
        break;

      case 'back':
        agent.velocity.x = -4.0;
        break;

      case 'left':
        agent.velocity.z = -3.0;
        break;

      case 'right':
        agent.velocity.z = 3.0;
        break;

      case 'jump':
        if (this.isGrounded && this.canJump) {
          // Jump velocity for realistic arc
          // v = sqrt(2 * g * h) where h is desired jump height
          // For a 1.5m jump at g=9.81: v = sqrt(2 * 9.81 * 1.5) = 5.4 m/s
          const desiredHeight = 2.0; // meters
          const jumpVelocity = Math.sqrt(2 * this.config.gravity * desiredHeight);
          agent.velocity.y = jumpVelocity;
          this.canJump = false;

          // Allow jump again after short delay (handled by grounded check)
        }
        break;

      case 'idle':
      default:
        // Apply some drag to horizontal velocity
        agent.velocity.x *= 0.9;
        agent.velocity.z *= 0.9;
        break;
    }

    // Re-enable jump when grounded
    if (this.isGrounded) {
      this.canJump = true;
    }

    // Clamp horizontal velocity
    const maxSpeed = 5.0;
    agent.velocity.x = Math.max(-maxSpeed, Math.min(maxSpeed, agent.velocity.x));
    agent.velocity.z = Math.max(-maxSpeed, Math.min(maxSpeed, agent.velocity.z));
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

    // Optional: Small shaping reward based on progress
    // Uncomment for shaped rewards:
    // const agent = this.world.getBody('agent');
    // if (agent) {
    //   const progress = (agent.position.x - (-this.platformWidth/4)) / (this.goalZone.minX - (-this.platformWidth/4));
    //   return Math.max(0, Math.min(0.1, progress * 0.1));
    // }

    return 0.0;
  }
}
