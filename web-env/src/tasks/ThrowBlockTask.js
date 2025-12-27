/**
 * ThrowBlockTask - Agent must pick up a block and throw it into a basket
 *
 * Actions: forward, back, left, right, pick, drop, throw_weak, throw_medium, throw_strong, idle
 * Success: Block rests inside basket at episode end
 */

import { BaseTask } from './BaseTask.js';

export class ThrowBlockTask extends BaseTask {
  constructor(world, config) {
    super(world, config);

    // Task-specific config
    this.basketDistance = config.basketDistance ?? 6.0;
    this.basketDistanceVariance = config.basketDistanceVariance ?? 1.0;
    this.basketHeight = config.basketHeight ?? 1.5;

    // Platform
    this.platformSize = 8.0;
    this.platformHeight = 0.5;

    // Block
    this.blockSize = 0.4;
    this.blockMass = 2.0;

    // Basket
    this.basketWidth = 1.2;
    this.basketDepth = 1.2;
    this.basketWallHeight = 0.8;
    this.basketWallThickness = 0.1;

    // Agent
    this.agentRadius = 0.4;
    this.agentHeight = 1.8;
    this.agentMass = 70;

    // State
    this.holdingBlock = false;
    this.isGrounded = false;

    // Throw strengths (impulse multipliers)
    this.throwStrengths = {
      weak: 5,
      medium: 10,
      strong: 15
    };

    // Action mapping
    this.actions = [
      'forward', 'back', 'left', 'right',
      'pick', 'drop',
      'throw_weak', 'throw_medium', 'throw_strong',
      'idle'
    ];

    this.setup();
  }

  setup() {
    // Calculate actual basket distance with variance
    const actualDistance = this.basketDistance +
      this.randomInRange(-this.basketDistanceVariance, this.basketDistanceVariance);

    // Platform
    this.world.createBox('platform', {
      position: [0, -this.platformHeight / 2, 0],
      size: [this.platformSize, this.platformHeight, this.platformSize],
      mass: 0
    });

    // Agent
    this.world.createCapsule('agent', {
      position: [-2, this.agentHeight / 2 + 0.1, 0],
      radius: this.agentRadius,
      height: this.agentHeight,
      mass: this.agentMass
    });

    // Block (near agent's feet)
    this.world.createBox('block', {
      position: [-1.5, this.blockSize / 2 + 0.1, 0],
      size: [this.blockSize, this.blockSize, this.blockSize],
      mass: this.blockMass
    });

    // Basket - create as multiple walls + floor
    const basketX = actualDistance;
    const basketY = this.basketHeight;

    // Basket floor
    this.world.createBox('basketFloor', {
      position: [basketX, basketY, 0],
      size: [this.basketWidth, this.basketWallThickness, this.basketDepth],
      mass: 0
    });

    // Basket walls (4 sides)
    const wallOffset = (this.basketWidth - this.basketWallThickness) / 2;
    const wallHeight = this.basketWallHeight;

    this.world.createBox('basketWallLeft', {
      position: [basketX - wallOffset, basketY + wallHeight / 2, 0],
      size: [this.basketWallThickness, wallHeight, this.basketDepth],
      mass: 0
    });

    this.world.createBox('basketWallRight', {
      position: [basketX + wallOffset, basketY + wallHeight / 2, 0],
      size: [this.basketWallThickness, wallHeight, this.basketDepth],
      mass: 0
    });

    this.world.createBox('basketWallFront', {
      position: [basketX, basketY + wallHeight / 2, -wallOffset],
      size: [this.basketWidth, wallHeight, this.basketWallThickness],
      mass: 0
    });

    this.world.createBox('basketWallBack', {
      position: [basketX, basketY + wallHeight / 2, wallOffset],
      size: [this.basketWidth, wallHeight, this.basketWallThickness],
      mass: 0
    });

    // Store basket bounds for detection
    this.basketBounds = {
      minX: basketX - this.basketWidth / 2 + this.basketWallThickness,
      maxX: basketX + this.basketWidth / 2 - this.basketWallThickness,
      minY: basketY,
      maxY: basketY + this.basketWallHeight,
      minZ: -this.basketDepth / 2 + this.basketWallThickness,
      maxZ: this.basketDepth / 2 - this.basketWallThickness
    };

    this.actualBasketDistance = actualDistance;
  }

  getObservation() {
    const agentState = this.world.getBodyState('agent');
    const blockState = this.world.getBodyState('block');

    return {
      // Agent state
      agentPosition: agentState.position,
      agentVelocity: agentState.velocity,

      // Block state
      blockPosition: blockState.position,
      blockVelocity: blockState.velocity,
      holdingBlock: this.holdingBlock,

      // Basket info
      basketPosition: [this.actualBasketDistance, this.basketHeight, 0],
      basketBounds: this.basketBounds,

      // Physics info
      gravity: this.config.gravity,
      isGrounded: this.isGrounded,

      // Available actions
      actions: this.actions
    };
  }

  applyAction(action) {
    const agent = this.world.getBody('agent');
    const block = this.world.getBody('block');
    if (!agent || !block) return;

    // Check if grounded
    this.isGrounded = Math.abs(agent.velocity.y) < 0.1 && agent.position.y < this.agentHeight / 2 + 0.3;

    // Normalize action
    const actionStr = typeof action === 'number' ? this.actions[action] : action.toLowerCase();

    switch (actionStr) {
      case 'forward':
        agent.velocity.x = 3.0;
        break;

      case 'back':
        agent.velocity.x = -3.0;
        break;

      case 'left':
        agent.velocity.z = -3.0;
        break;

      case 'right':
        agent.velocity.z = 3.0;
        break;

      case 'pick':
        if (!this.holdingBlock) {
          // Check if close enough to block
          const dx = agent.position.x - block.position.x;
          const dz = agent.position.z - block.position.z;
          const dist = Math.sqrt(dx * dx + dz * dz);

          if (dist < 1.5) {
            this.holdingBlock = true;
          }
        }
        break;

      case 'drop':
        if (this.holdingBlock) {
          this.holdingBlock = false;
          // Place block at agent's position
          block.position.set(
            agent.position.x + 0.5,
            agent.position.y - 0.5,
            agent.position.z
          );
          block.velocity.set(0, 0, 0);
        }
        break;

      case 'throw_weak':
      case 'throw_medium':
      case 'throw_strong':
        if (this.holdingBlock) {
          this.holdingBlock = false;

          // Get throw strength
          const strengthKey = actionStr.replace('throw_', '');
          const strength = this.throwStrengths[strengthKey];

          // Calculate throw direction (forward, slightly up)
          // Adjust for gravity - weaker gravity needs less vertical impulse
          const gravityFactor = this.config.gravity / 9.81;

          // Position block in front of agent
          block.position.set(
            agent.position.x + 0.8,
            agent.position.y + 0.3,
            agent.position.z
          );

          // Apply throw impulse
          // Horizontal velocity scales with strength
          // Vertical velocity scales with gravity factor
          block.velocity.set(
            strength,                          // Forward
            strength * 0.5 * Math.sqrt(gravityFactor), // Up (adjusted for gravity)
            0                                   // Sideways
          );
        }
        break;

      case 'idle':
      default:
        agent.velocity.x *= 0.9;
        agent.velocity.z *= 0.9;
        break;
    }

    // If holding block, move block with agent
    if (this.holdingBlock) {
      block.position.set(
        agent.position.x + 0.5,
        agent.position.y + 0.2,
        agent.position.z
      );
      block.velocity.set(0, 0, 0);
    }

    // Clamp agent velocity
    const maxSpeed = 4.0;
    agent.velocity.x = Math.max(-maxSpeed, Math.min(maxSpeed, agent.velocity.x));
    agent.velocity.z = Math.max(-maxSpeed, Math.min(maxSpeed, agent.velocity.z));
  }

  checkTermination() {
    const block = this.world.getBody('block');
    const agent = this.world.getBody('agent');

    if (!block || !agent) {
      return { done: true, success: false, reason: 'missing_objects' };
    }

    // Check if block fell off platform
    if (block.position.y < -2) {
      return { done: true, success: false, reason: 'block_fell' };
    }

    // Check if agent fell off platform
    if (agent.position.y < -2) {
      return { done: true, success: false, reason: 'agent_fell' };
    }

    // Check if block is in basket and at rest
    const pos = block.position;
    const vel = block.velocity;
    const speed = Math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);

    const inBasket =
      pos.x >= this.basketBounds.minX && pos.x <= this.basketBounds.maxX &&
      pos.y >= this.basketBounds.minY && pos.y <= this.basketBounds.maxY &&
      pos.z >= this.basketBounds.minZ && pos.z <= this.basketBounds.maxZ;

    const atRest = speed < 0.5;

    if (inBasket && atRest) {
      this.successSteps++;
      if (this.successSteps >= this.requiredSuccessSteps) {
        return { done: true, success: true, reason: 'block_in_basket' };
      }
    } else {
      this.successSteps = 0;
    }

    return { done: false, success: false, reason: 'ongoing' };
  }

  getReward(done, success) {
    return success ? 1.0 : 0.0;
  }
}
