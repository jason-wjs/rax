// Keep tool images visible to Messages gateways that stringify nested tool content.
export function liftToolResultImages(body) {
  for (const message of body.messages ?? []) {
    if (message.role !== 'user' || !Array.isArray(message.content)) continue;
    const lifted = [];
    for (const block of message.content) {
      if (block.type !== 'tool_result' || !Array.isArray(block.content)) continue;
      const images = block.content.filter(item => item.type === 'image');
      if (!images.length) continue;
      block.content = block.content.filter(item => item.type !== 'image');
      if (!block.content.length) block.content.push({ type: 'text', text: 'Tool images attached below.' });
      lifted.push({ type: 'text', text: `Images from tool result ${block.tool_use_id}:` }, ...images);
    }
    message.content.push(...lifted);
  }
  return body;
}
