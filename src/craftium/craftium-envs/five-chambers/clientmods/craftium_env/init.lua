-- Five Chambers client mod.
-- Rewards are delivered server-side via craftium.reward() in state_files.lua.
-- This mod only handles soft-reset signalling over the mod channel.

local channel_name = "craftium_channel"
local mod_channel  = nil

-- Forward-declared so the two functions can reference each other.
local callback
local wait_for_client_ready

wait_for_client_ready = function()
    if minetest.localplayer then
        callback()
    else
        minetest.after(0.01, wait_for_client_ready)
    end
end

callback = function()
    mod_channel = minetest.mod_channel_join(channel_name)
    local my_name = minetest.localplayer:get_name()
    print("[five_chambers client] " .. my_name .. " joined channel " .. channel_name)
end

-- Send soft-reset signal to server when craftium requests an episode reset.
minetest.register_globalstep(function(dtime)
    if get_soft_reset() == 1 and mod_channel ~= nil then
        mod_channel:send_all(minetest.serialize({
            agent = "server",
            reset = true,
        }))
        reset_termination()
    end
end)

wait_for_client_ready()
